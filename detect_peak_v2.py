"""
A dense neural network based algorithm to assist astronomers in analysing broad spectral features in IFU datasets.
This algorithm is desighned based on MUSE NFM AO mode data cube. Look for instructions inline to adapt for your desired instrument.
The algorithm simulates broad spectral features (upto 2) in this version. The neural architecture is denisgned as 128-dropout-64-dropout-output head. 
This code features two output heads-- classification head and regression head which is used to detect the number of peaks in the spectrum slice and the position of these peaks.
Guided by the neural network output, the algorithm fits a two or single gaussian profile on the spectrum. 

This program is designed by Priyam Das. email: priyam.das@unsw.edu.au
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit
import json
import pyfiglet
from termcolor import colored
# -----------------------------
# 1. Helper Functions
# -----------------------------
# Builds a 1D pixel grid with similar configuration as MUSE WFM AO mode. Tweak this to suite your instrument. 
# 3722 is the total number of pixel in the spectrum
# 4699.95654296875 is the starting wavelength at pixel 1.
# 1.25 is the step
t = np.array([4699.95654296875 + i * 1.25 for i in range(1, 3722)])  
d = 2.7e-4  # scale factor (e.g. arcsec per pixel)




# -----------------------------
# 2. Define Gaussian Functions for Fitting
# -----------------------------

def gaussian(x,m,c, A, mu, sigma):
    """Single Gaussian function.
    m and c are the slope and intercept of the linear continuum. A is the amplitude of the Gaussian, 
    mu is the mean (peak position), and sigma is the standard deviation (width) of the Gaussian.
    The function returns the value of the Gaussian plus the linear continuum at each x.
    """
    return m*x+c+A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def double_gaussian(x,m,c, A1, mu1, sigma1, A2, mu2, sigma2):
    """Sum of two Gaussian functions."""
    return m*x +c+(A1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2))) +(A2 * np.exp(-(x - mu2)**2 / (2 * sigma2**2)))

# -----------------------------
# 3. Simulate and Prepare Data
# -----------------------------

# Create wavelength array covering the full range (used for plotting and fitting)


# For our application, we want to focus on the region around 5303 A and 7600 A.
# Here we extract indices corresponding to 5200 to 5548 A and 7600 to 7510 A.
# The wavelength range is read from the params.json files. Prepare your own .json file to define your simulation parameters.
# An Example .json file is provided for reference.

# Function to simulate a spectrum based on a given class:
# 0: no peak, 1: one broad peak, 2: two broad peaks
def simulate_spectrum(class_label, xmin, xmax, 
                      lmin, lmax, l, 
                      A, A1, A2, n):
    """
    Simulate a spectrum based on the specified class label and parameters. 
    The function generates a spectrum with either no peak, one broad peak, or two broad peaks, depending on the class_label input. The parameters allow for randomization of the peak characteristics to create a diverse training dataset for the neural network.
    Parameters
    ----------
    class_label: int
        0 for no peak, 1 for one broad peak, 2 for two broad peaks
    xmin, xmax: int
        Wavelength range for the spectrum (in pixel indices)
    lmin, lmax: float
        Wavelength range for peak position randomization (in Angstroms)
    l: float
        Rest wavelength of the line (in Angstroms)
    A, A1, A2: float
        Amplitude parameters for the peaks (in arbitrary units)
    n: float
        Noise level (standard deviation of the Gaussian noise to be added to the spectrum)
    
    Returns
    -------
    spec: array
        Simulated spectrum with the specified characteristics
    peak_pos: list
    
    """
    indices = np.arange(xmin, xmax)
    spec = np.zeros_like(t[indices])
    peak_pos = [-1.0, -1.0]  # Default peak positions for class 0

    if class_label == 1:
        A = np.random.choice(np.arange(A-5, A+10, 1), size=1)[0]        # Randomized peak amplitude based on actual amplitude ranges.
        m = np.random.choice(np.arange(-0.05, 0.01, 0.01), size=1)[0]   # Randomized slope parameters to fit various continuum
        s = np.random.choice(np.arange(18, 40, 0.5), size=1)[0]         # Randomized sigma values to reflect varying velocity width in actual data. ******** Change to 12-30 for Fe XIV.
        mu = np.random.choice(np.arange(l-100, l+50, 0.5), size=1)[0]   # Randomized occurance of peak in the spectrum for robust classification
        spec += gaussian(t[indices], m, 3, A, mu=mu, sigma=s)
        spec += np.random.normal(0, n, size=spec.shape)                 # Adding noise to simulate real data
        peak_pos = [mu, -10.0]

    elif class_label == 2:
        A1 = np.random.choice(np.arange(A1-20, A1+15, 1), size=1)[0]    # Randomized ampltitude
        A2 = np.random.choice(np.arange(A2-15, A2+10, 1), size=1)[0]
        sigma1 = np.random.choice(np.arange(12, 30, 0.5), size=1)[0]    # Randomized sigma
        sigma2 = np.random.choice(np.arange(12, 30, 0.5), size=1)[0]
        m = np.random.choice(np.arange(-0.07, 0.07, 0.01), size=1)[0]
        mu1 = np.random.choice(np.arange(lmin, l-7, 0.5), size=1)[0]    # Randomized peak position. However, we employ a separation window of 15 A so that the two peaks may
        mu2 = np.random.choice(np.arange(l+8, lmax, 0.5), size=1)[0]    # blend but are always distinct - reflecting the real dataset. 
        spec += double_gaussian(t[indices], m, 5, A1, mu1, sigma1, A2, mu2, sigma2)
        spec += np.random.normal(0, n, size=spec.shape)
        peak_pos = [mu1, mu2]

    elif class_label == 0:
        m = np.random.choice(np.arange(-0.09, 0, 0.01), size=1)[0]
        spec += m * t[indices] + 5 + np.random.normal(0, n, size=spec.shape)
        # peak_pos remains [-1, -1]

    return spec, peak_pos




# -----------------------------
# 5. Peak Detection and Gaussian Fitting Function
"""
This function fits a gaussian curve based on the output from the neural network. A single or double gaussian fit is attempted for a class 1 and class 2 spectrum respectively.
The initial guesses for the curve_fit depends on the predicted peak position. Sanity checks are performed on the fit to make sure the Gaussian curves are valid for our science.
If sanity checks fail, a series of initial guess for the peak is considered with gradual shifts of 25 pixel in the peak position.
"""
# -----------------------------
def gaussianx(x,m,c, A, mu, sigma):
    """Single Gaussian function."""
    return m*x+c+(A * np.exp(-0.5 * ((x - mu) / sigma) ** 2))

def double_gaussianx(x,m,c, A1, mu1, sigma1, A2, mu2, sigma2):
    """Sum of two Gaussian functions."""
    return m*x +c+(A1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2))) +(A2 * np.exp(-(x - mu2)**2 / (2 * sigma2**2)))
def detect_and_fit(spectrum, model, x_vals, lambda_max, lambda_min, s=5315,mu_h1=5225,sigma1=15,mu_h2=5365,sigma2=15):
    """
    Given a spectrum and the trained model:
      - Standardize the spectrum.
      - Predict its class (0: no peak, 1: one peak, 2: two peaks).
      - Predict the peak position which serves as initial guess
      - If peaks are detected, fit a Gaussian curve(s).
    The function includes sanity checks on the fitted parameters and implements a fallback mechanism to ensure robust peak detection and fitting, 
    even in cases where the initial fit may not be ideal.

    Parameters
    ----------
    spectrum: array
        The input spectrum to be analyzed.
    model: Keras model
        The trained neural network model for classification and regression.
    x_vals: array
        The wavelength array corresponding to the spectrum.
    lambda_max, lambda_min: float
        The maximum and minimum wavelength values for scaling the predicted peak positions.
    s: float
        Rest wavelength of the line (in Angstroms) - used for sanity checks on peak positions.
    mu_h1, sigma1: float
        Expected mean and standard deviation for the first peak (used for sanity checks).
    mu_h2, sigma2: float
        Expected mean and standard deviation for the second peak (used for sanity checks).
    
    Returns
    -------
    popt: array
        Optimal parameters for the fitted Gaussian(s).
    fit_curve: array
        The fitted Gaussian curve(s) evaluated at x_vals.
    
    """
    # Standardize the input spectrum
    spec_std = (spectrum - np.mean(spectrum)) / (np.std(spectrum) if np.std(spectrum) != 0 else 1)
    spec_std = spec_std.reshape(1, -1)
    pred_probs, pred_peaks = model.predict(spec_std)
    pred_class = np.argmax(pred_probs[0])
    scaled_peaks = pred_peaks[0] * (lambda_max - lambda_min) + lambda_min
    peak1, peak2 = scaled_peaks # Extract predicted peak positions
    #amp1 = np.mean(spectrum[np.argmin(np.abs(x_vals - peak2))-5:np.argmin(np.abs(x_vals - peak2))+5])
    #amp2= np.mean(spectrum[np.argmin(np.abs(x_vals - peak1))-5:np.argmin(np.abs(x_vals - peak1))+5])
    #tqdm.write(f"Predicted class: {pred_class} (Probabilities: {pred_probs})")
    
    
    # Depending on the predicted class, fit the appropriate Gaussian(s)
    if pred_class == 1:
        tqdm.write(f"Predicted class: {pred_class} (Network peak1: {peak1}, Network peak 2: {peak2})")

        # helper to get amplitude at nearest wavelength
        def amp_at(mu):
            idx = np.argmin(np.abs(x_vals - mu))
            return float(spectrum[idx])

        def valid_single(popt, sigma_min=15.0):
            # expect popt = [c0, c1, A, mu, sigma]
            if popt is None or len(popt) < 5:
                return False
            A, mu, sig = float(popt[2]), float(popt[3]), float(popt[4])
            if A <= 0 or sig < sigma_min:
                return False
            if not (x_vals.min() <= mu <= x_vals.max()):
                return False
            return True

        # trial centers and order requested
        trials = [peak1, peak1 - 25, peak1 - 50, peak1 + 25, peak1 + 50]

        popt = None
        fit_curve = None

        lower_bounds = [-np.inf, -np.inf, 0.0, x_vals.min(), 10.0]
        upper_bounds = [np.inf, np.inf, np.inf, x_vals.max(), 40.0]

        for mu_try in trials:
            # skip out-of-range trial mus
            if not (x_vals.min() <= mu_try <= x_vals.max()):
                tqdm.write(f"Skipping out-of-range trial mu: {mu_try}")
                continue

            p0 = [0.05, 10.0, amp_at(mu_try), mu_try, 20.0]
            try:
                popt_try, _ = curve_fit(gaussianx, x_vals, spectrum, p0=p0,
                                        bounds=(lower_bounds, upper_bounds), maxfev=5000)
                fit_try = gaussianx(x_vals, *popt_try)
                if np.abs(popt_try[0]) > 6e-3:                              #****Change slope value for your data.***********

                    tqdm.write("Slope too high — flattening spectrum and refitting at maximum.")

                    # Linear detrending
                    coeff = np.polyfit(x_vals, spectrum, 1)
                    linear_fit = np.polyval(coeff, x_vals)

                    # Prevent division instability
                    linear_fit[linear_fit == 0] = 1e-8

                    flat_spectrum = spectrum / linear_fit

                    # Fit at maximum of flattened spectrum
                    idx_max = np.argmax(flat_spectrum)
                    mu_init = x_vals[idx_max]

                    p0_flat = [0.0, 0.0, flat_spectrum[idx_max], mu_init, 20.0]

                    popt_flat, _ = curve_fit(
                        gaussianx,
                        x_vals,
                        flat_spectrum,
                        p0=p0_flat,
                        bounds=(lower_bounds, upper_bounds),
                        maxfev=5000
                    )

                    # Refit original spectrum using refined μ and σ
                    mu_refined = popt_flat[3]
                    sigma_refined = np.clip(popt_flat[4], 15.0, 80.0)

                    amp_refined = spectrum[np.argmin(np.abs(x_vals - mu_refined))]

                    p0_refit = [0.05, 0.0, amp_refined, mu_refined, sigma_refined]

                    popt_refit, _ = curve_fit(
                        gaussianx,
                        x_vals,
                        spectrum,
                        p0=p0_refit,
                        bounds=(lower_bounds, upper_bounds),
                        maxfev=5000
                    )

                    fit_curve = gaussianx(x_vals, *popt_refit)
                    if valid_single(popt_refit, sigma_min=10.0):
                        popt, fit_curve = popt_refit, fit_curve
                        tqdm.write(f"Flattened single-gaussian accepted: {popt}")
                        break
                    else:
                        tqdm.write("Flattened refit failed sanity check.")
                elif valid_single(popt_try):
                    popt, fit_curve = popt_try, fit_try
                    tqdm.write(f"Accepted single-gaussian fit at mu={mu_try}: {popt}")
                    break
                else:
                    tqdm.write(f"Rejected single fit at mu={mu_try} (failed sanity checks): {popt_try}")
            except RuntimeError:
                tqdm.write(f"Single fit at mu={mu_try} raised RuntimeError")

        # If all shifted trials failed, do final fallback: fit at global max position
        if popt is None:
            tqdm.write("All shifted single-gaussian attempts failed — trying fallback at spectrum maximum.")

            idx_max = np.argmax(spectrum)
            mu_init = x_vals[idx_max]

            # --- Linear detrending ---
            coeff = np.polyfit(x_vals, spectrum, 1)
            linear_fit = np.polyval(coeff, x_vals)

            # Avoid division instability
            linear_fit[linear_fit == 0] = 1e-8
            flat_spectrum = spectrum / linear_fit

            # --- Step 1: Fit flattened spectrum ---
            p0_flat = [0.05, 4.0, float(flat_spectrum[idx_max]), mu_init, 15.0]

            try:
                popt_flat, _ = curve_fit(
                    gaussianx,
                    x_vals,
                    flat_spectrum,
                    p0=p0_flat,
                    bounds=(lower_bounds, upper_bounds),
                    maxfev=5000
                )

                # --- Step 2: Refit original spectrum using μ and σ from flattened fit ---
                mu_refined = popt_flat[3]
                sigma_refined = max(popt_flat[4], 15.0)

                amp_refined = spectrum[np.argmin(np.abs(x_vals - mu_refined))]

                p0_refit = [0.05, 30.0, amp_refined, mu_refined, sigma_refined]

                popt_refit, _ = curve_fit(
                    gaussianx,
                    x_vals,
                    spectrum,
                    p0=p0_refit,
                    bounds=(lower_bounds, upper_bounds),
                    maxfev=5000
                )

                fit_curve = gaussianx(x_vals, *popt_refit)

                # --- Final validation ---
                if valid_single(popt_refit, sigma_min=10.0):
                    popt = popt_refit
                    tqdm.write(f"Fallback single-gaussian accepted at max: {popt}")
                else:
                    tqdm.write(f"Fallback single-gaussian failed sanity check: {popt_refit}")
                    popt = None

            except RuntimeError:
                tqdm.write("Fallback single-gaussian raised RuntimeError")
                popt = None

        # Final outcome
        if popt is not None:
            return pred_class, popt, fit_curve
        else:
            tqdm.write("All single-gaussian fitting attempts failed. Returning class=0, no fit.")
            pred_class = 0
        return pred_class, None, None
        
    elif pred_class == 2:
        tqdm.write(f"Predicted class: {pred_class} (Peak positions: {peak1}, {peak2})")

        # ensure peak1 <= peak2
        peak1, peak2 = np.sort([peak1, peak2])

        # helper: amplitude at nearest pixel
        def amp_at(mu):
            amp = np.mean(spectrum[np.argmin(np.abs(x_vals - mu))-5:np.argmin(np.abs(x_vals - mu))+5])
            return float(amp)

        # sanity check for a fitted double-gaussian parameter vector popt
        def valid_double(popt, s_ref, sigma_min=11.0):
            try:
                # expect at least 8 parameters: c0, c1, A1, mu1, sigma1, A2, mu2, sigma2
                if len(popt) < 8:
                    return False
                A1, mu1, sig1 = float(popt[2]), float(popt[3]), float(popt[4])
                A2, mu2, sig2 = float(popt[5]), float(popt[6]), float(popt[7])
                # amplitudes positive, sigmas above threshold
                if not (A1 > 0 and A2 > 0 and sig1 >= sigma_min and sig2 >= sigma_min):
                    return False
                # peaks should lie on opposite sides of s_ref (one < s_ref, one > s_ref)
                if not ((mu1 < s_ref and mu2 > s_ref) or (mu1 > s_ref and mu2 < s_ref)):
                    return False
                # mus should lie within x_vals range
                if not (x_vals.min() <= mu1 <= x_vals.max() and x_vals.min() <= mu2 <= x_vals.max()):
                    return False
                return True
            except Exception:
                return False

        # initial amplitudes and sigmas
        amp1_guess = amp_at(peak1)
        amp2_guess = amp_at(peak2)
        sigma1 = sigma1
        sigma2 = sigma2

        # prepare p0 for initial attempt
        p0_initial = [0.05, 1.0, amp1_guess, peak1, sigma1, amp2_guess, peak2, sigma2]

        # bounds: keep mus within wavelength array and sigmas positive
        lower_bounds = [-np.inf, -np.inf, 0.0, x_vals.min(), 10.0, 0.0, x_vals.min(), 10]
        upper_bounds = [np.inf, np.inf, np.inf, x_vals.max(), 40.0, np.inf, x_vals.max(), 40.0]

        popt = None
        fit_curve = None

        # try initial fit
        try:
            popt_try, _ = curve_fit(double_gaussianx, x_vals, spectrum, p0=p0_initial,
                                    bounds=(lower_bounds, upper_bounds), maxfev=5000)
            fit_curve_try = double_gaussianx(x_vals, *popt_try)
            if valid_double(popt_try, s):
                popt, fit_curve = popt_try, fit_curve_try
                tqdm.write(f"Initial double fit accepted: {popt}")
            else:
                tqdm.write(f"Initial double fit failed sanity checks: {popt_try}")
        except RuntimeError:
            tqdm.write("Initial double fit raised RuntimeError")

        # If initial fit failed, try the requested shifted guesses (in this exact order)
        if popt is None:
            shifts = [
                (peak1, peak2 - 25),
                (peak1 - 25, peak2),
                (peak1 - 25, peak2 - 25),
                (peak1, peak2 + 25),
                (peak1 + 25, peak2),
                (peak1 + 25, peak2 + 25),
                (peak1-50,peak2),
                (peak1, peak2-50),
                (peak1-50,peak2-50)
            ]

            for mu1_try, mu2_try in shifts:
                # ensure trial mus remain inside wavelength bounds
                if not (x_vals.min() <= mu1_try <= x_vals.max() and x_vals.min() <= mu2_try <= x_vals.max()):
                    tqdm.write(f"Skipping out-of-range trial mus: {mu1_try}, {mu2_try}")
                    continue

                p0 = [0.05, 1.0,
                    amp_at(mu1_try), mu1_try, 20.0,
                    amp_at(mu2_try), mu2_try, 20.0]
                try:
                    popt_try, _ = curve_fit(double_gaussianx, x_vals, spectrum, p0=p0,
                                            bounds=(lower_bounds, upper_bounds), maxfev=5000)
                    fit_curve_try = double_gaussianx(x_vals, *popt_try)
                    if valid_double(popt_try, s):
                        popt, fit_curve = popt_try, fit_curve_try
                        tqdm.write(f"Accepted shifted double fit with mus ({mu1_try}, {mu2_try}): {popt}")
                        break
                    else:
                        tqdm.write(f"Shifted fit ({mu1_try}, {mu2_try}) failed sanity checks: {popt_try}")
                except RuntimeError:
                    tqdm.write(f"Shifted fit ({mu1_try}, {mu2_try}) raised RuntimeError and was skipped")

        # If none of the shifted fits succeeded, try the hardcoded fallback double fit you used before
        if popt is None:
            p0_hard = [0.05, 1.0, np.max(spectrum), mu_h1, 15.0,
                    0.2, mu_h2, 24.0]
            try:
                popt_try, _ = curve_fit(double_gaussianx, x_vals, spectrum, p0=p0_hard,
                                        bounds=(lower_bounds, upper_bounds), maxfev=5000)
                fit_curve_try = double_gaussianx(x_vals, *popt_try)
                if valid_double(popt_try, s):
                    popt, fit_curve = popt_try, fit_curve_try
                    tqdm.write(f"Accepted hardcoded double fit: {popt}")
                else:
                    tqdm.write(f"Hardcoded double fit failed sanity checks: {popt_try}")
            except RuntimeError:
                tqdm.write("Hardcoded double fit raised RuntimeError")

        # If still no valid double fit, try single-Gaussian fallback
        if popt is None:
            tqdm.write("All double-gaussian attempts failed — trying single-gaussian fallback.")
            idx_max = np.argmax(spectrum)
            mu_init = x_vals[idx_max]
            p0_single = [0.05, 4.0, spectrum[idx_max], mu_init, 15.0]

            try:
                popt_single, _ = curve_fit(gaussianx, x_vals, spectrum, p0=p0_single, maxfev=5000)
                fit_curve_single = gaussianx(x_vals, *popt_single)
                # basic validation for single peak: positive amp and sigma
                if popt_single[2] > 0 and popt_single[4] > 1.0:
                    # convert single fit to "double-like" vector by padding with NaNs so caller sees 8 params
                    popt = np.concatenate([popt_single, np.array([np.nan, np.nan, np.nan])])
                    fit_curve = fit_curve_single
                    pred_class = 1
                    tqdm.write(f"Single-gaussian accepted and returned (as class=1): {popt_single}")
                else:
                    tqdm.write(f"Single-gaussian failed basic checks: {popt_single}")
            except RuntimeError:
                tqdm.write("Single-gaussian fallback raised RuntimeError")

        # Final safety: if everything failed, return NaN-padded vector and fit_curve = None
        if popt is None:
            tqdm.write("All fitting attempts failed. Returning NaN-padded parameters.")
            popt = np.full(8, np.nan)
            fit_curve = None
            pred_class = 0

        return pred_class, popt, fit_curve
    else:
        tqdm.write("No peak detected.")
        return pred_class, None, None




######################## MAIN EXECUTION FILE ###############################
"""
Read the necessary parameters from the .json file and execute the ML and curve fit codes.
"""

def implement_inn(params_file, reg1, model, plot_spectrum=True, fit_plot=True):
    """
    This function serves as the main execution point for the peak detection and fitting process. 
    It reads the necessary parameters from a specified .json file, extracts an example spectrum from the provided data, 
    and then utilizes the trained neural network model to predict the class and peak positions. Based on the predictions, 
    it attempts to fit Gaussian curves to the spectrum and includes options for plotting the original spectrum and the fitted curves for visual verification.

    Parameters
    ----------
    params_file: str
        Path to the .json file containing simulation and fitting parameters.
    reg1: array
        The data array from which the example spectrum will be extracted.
    model: Keras model
        The trained neural network model for classification and regression.
    plot_spectrum: bool
        If True, plots the example spectrum before fitting.
    fit_plot: bool
        If True, plots the original spectrum along with the fitted Gaussian curve(s) after fitting.
    
    Returns
    -------
    pred_class: int
        The predicted class of the spectrum (0: no peak, 1: one peak, 2: two peaks).
    popt: array
        Optimal parameters for the fitted Gaussian(s).
    fit_curve: array
        The fitted Gaussian curve(s) evaluated at the x_vals.   
    """
    with open(params_file, 'r') as file:
            data = json.load(file)
    min_pixel = data["Simulated_spectra_parameters"]["xmin"]
    max_pixel = data["Simulated_spectra_parameters"]["xmax"]
    s=data["Actual_model_fit_parameters"]["Separation_lambda"]
    amp1=data["Actual_model_fit_parameters"]["Amplitude1"]
    mu1=data["Actual_model_fit_parameters"]["mu1"]
    sigma1=data["Actual_model_fit_parameters"]["sigma1"]
    amp2=data["Actual_model_fit_parameters"]["Amplitude2"]
    lambda_max = data["Simulated_spectra_parameters"]["lambda_min"]
    lambda_min = data["Simulated_spectra_parameters"]["lambda_max"]
    mu2=data["Actual_model_fit_parameters"]["mu2"]
    sigma2=data["Actual_model_fit_parameters"]["sigma2"]
    name=data["Species"]
    big_text = pyfiglet.figlet_format(f"Computing velocity map for {name}...")
    tqdm.write(colored(big_text, "red", attrs=["bold"]))
    
    
    #reg1, data = sum_region(fits_file, x_coords, y_coords, sum=False)
    example_spectrum = reg1[min_pixel:max_pixel]

    

    if plot_spectrum:
        plt.plot(t[min_pixel:max_pixel], example_spectrum)
        plt.xlabel('Wavelength (Angstroms)')
        plt.ylabel('Flux')
        plt.title('Example Spectrum')
        plt.show()

    x_vals = t[min_pixel:max_pixel]

    pred_class, popt, fit_curve = detect_and_fit(example_spectrum, model, x_vals,lambda_max,lambda_min,s,mu1,sigma1,mu2,sigma2)

    # Plot the original spectrum and the fitted Gaussian(s) if available.
    if fit_plot:
        plt.figure(figsize=(10, 5))
        plt.plot(x_vals, example_spectrum, label='Original Spectrum')
        if fit_curve is not None:
            plt.plot(x_vals, fit_curve, label='Gaussian Fit', linestyle='--')
        plt.xlabel('Wavelength (Å)')
        plt.ylabel('Flux')
        plt.legend()
        plt.title(f"Detection Result: Class {pred_class}")
        plt.show()
    return pred_class, popt, fit_curve

