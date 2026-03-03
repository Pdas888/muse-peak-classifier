from xml.parsers.expat import model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization.wcsaxes import add_scalebar
from matplotlib.patches import Rectangle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report,confusion_matrix, accuracy_score
import tensorflow as tf
#from detect_star import analyze_clumps
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from scipy.optimize import curve_fit
import json
import pyfiglet
from termcolor import colored
# -----------------------------
# 1. Helper Functions
# -----------------------------
t = np.array([4699.95654296875 + i * 1.25 for i in range(1, 3722)])
d = 2.7e-4  # scale factor (e.g. arcsec per pixel)




# -----------------------------
# 2. Define Gaussian Functions for Fitting
# -----------------------------

def gaussian(x,m,c, A, mu, sigma):
    """Single Gaussian function."""
    return m*x+c+A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def double_gaussian(x,m,c, A1, mu1, sigma1, A2, mu2, sigma2):
    """Sum of two Gaussian functions."""
    return m*x +c+(A1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2))) +(A2 * np.exp(-(x - mu2)**2 / (2 * sigma2**2)))

# -----------------------------
# 3. Simulate and Prepare Data
# -----------------------------

# Create wavelength array covering the full range (used for plotting and fitting)


# For our application, we want to focus on the region around 5303 Å.
# Here we extract indices corresponding roughly to 5200 to 5548 Å.



# Function to simulate a spectrum based on a given class:
# 0: no peak, 1: one broad peak, 2: two broad peaks
def simulate_spectrum(class_label, xmin, xmax, 
                      lmin, lmax, l, 
                      A, A1, A2, n):
    indices = np.arange(xmin, xmax)
    spec = np.zeros_like(t[indices])
    peak_pos = [-1.0, -1.0]  # Default peak positions for class 0

    if class_label == 1:
        A = np.random.choice(np.arange(A-5, A+10, 1), size=1)[0]
        m = np.random.choice(np.arange(-0.05, 0.01, 0.01), size=1)[0]
        s = np.random.choice(np.arange(12, 30, 0.5), size=1)[0]
        mu = np.random.choice(np.arange(l-100, l+50, 0.5), size=1)[0]
        spec += gaussian(t[indices], m, 3, A, mu=mu, sigma=s)
        spec += np.random.normal(0, n, size=spec.shape)
        peak_pos = [mu, -1.0]

    elif class_label == 2:
        A1 = np.random.choice(np.arange(A1-20, A1+15, 1), size=1)[0]
        A2 = np.random.choice(np.arange(A2-15, A2+10, 1), size=1)[0]
        sigma1 = np.random.choice(np.arange(12, 30, 0.5), size=1)[0]
        sigma2 = np.random.choice(np.arange(12, 30, 0.5), size=1)[0]
        m = np.random.choice(np.arange(-0.07, 0.07, 0.01), size=1)[0]
        mu1 = np.random.choice(np.arange(lmin, l-7, 0.5), size=1)[0]
        mu2 = np.random.choice(np.arange(l+8, lmax, 0.5), size=1)[0]
        spec += double_gaussian(t[indices], m, 5, A1, mu1, sigma1, A2, mu2, sigma2)
        spec += np.random.normal(0, n, size=spec.shape)
        peak_pos = [mu1, mu2]

    elif class_label == 0:
        m = np.random.choice(np.arange(-0.09, 0, 0.01), size=1)[0]
        spec += m * t[indices] + 5 + np.random.normal(0, n, size=spec.shape)
        # peak_pos remains [-1, -1]

    return spec, peak_pos



# -----------------------------
# 4. Build and Train the Neural Network Model
# -----------------------------

# Build a fully connected neural network for 3-class classification





# -----------------------------
# 5. Peak Detection and Gaussian Fitting Function
# -----------------------------
def gaussianx(x,m,c, A, mu, sigma):
    """Single Gaussian function."""
    return m*x+c+(A * np.exp(-0.5 * ((x - mu) / sigma) ** 2))

def double_gaussianx(x,m,c, A1, mu1, sigma1, A2, mu2, sigma2):
    """Sum of two Gaussian functions."""
    return m*x +c+(A1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2))) +(A2 * np.exp(-(x - mu2)**2 / (2 * sigma2**2)))
def detect_and_fit(spectrum, model, x_vals, lambda_max, lambda_min, s=5315,amp1=0,mu1=5225,sigma1=15,amp2=0,mu2=5365,sigma2=15):
    """
    Given a spectrum and the trained model:
      - Standardize the spectrum.
      - Predict its class (0: no peak, 1: one peak, 2: two peaks).
      - If peaks are detected, fit a Gaussian curve(s).
    """
    # Standardize the input spectrum
    spec_std = (spectrum - np.mean(spectrum)) / (np.std(spectrum) if np.std(spectrum) != 0 else 1)
    spec_std = spec_std.reshape(1, -1)
    pred_probs, pred_peaks = model.predict(spec_std)
    pred_class = np.argmax(pred_probs[0])
    scaled_peaks = pred_peaks[0] * (lambda_max - lambda_min) + lambda_min
    peak1, peak2 = scaled_peaks # Extract predicted peak positions
    amp1 = np.mean(spectrum[np.argmin(np.abs(x_vals - peak2))-5:np.argmin(np.abs(x_vals - peak2))+5])
    amp2= np.mean(spectrum[np.argmin(np.abs(x_vals - peak1))-5:np.argmin(np.abs(x_vals - peak1))+5])
    #tqdm.write(f"Predicted class: {pred_class} (Probabilities: {pred_probs})")
    
    
    # Depending on the predicted class, fit the appropriate Gaussian(s)
    if pred_class == 1:
        
        tqdm.write(f"Predicted class: {pred_class} (Peak position: {peak1})")
        # Fit one-component Gaussian: initial guess from the data
        p0 = [0.05,30,amp2, peak1, 20]
        try:
            popt, _ = curve_fit(gaussianx, x_vals, spectrum, p0=p0)
            tqdm.write(f"Fitted one-component Gaussian parameters: {popt}")
            fit_curve = gaussianx(x_vals, *popt)
            if popt[3]<x_vals[np.argmin(x_vals+5)]:
                coeff= np.polyfit(x_vals, spectrum, 1)
                linear_fit = np.polyval(coeff, x_vals)
                flat_spectrum=spectrum/linear_fit
                p0 = [0.05,30,np.max(flat_spectrum), x_vals[np.argmax(flat_spectrum)], 20]
                popt, _ = curve_fit(gaussianx, x_vals, flat_spectrum, p0=p0)
                #fit_curve = gaussianx(x_vals, *popt)
                p0 = [0.05,30,100, popt[3], 20]
                popt, _ = curve_fit(gaussianx, x_vals, spectrum, p0=p0)
                fit_curve = gaussianx(x_vals, *popt) 
            attempt = 0
            max_attempts = 5  # Prevent infinite loop
            while (popt[4] < 15 or popt[2]<0) and attempt < max_attempts:
                tqdm.write(f"Peak is too narrow (sigma={popt[4]:.2f}). Re-fitting with shifted guess.")
                if attempt == 0:
                    p0 = [0.05, 30, np.max(spectrum), peak1-50, 20]
                elif attempt == 1:
                    p0 = [0.05, 30, np.max(spectrum), peak1-25, 20]
                elif attempt == 2:
                    p0 = [0.05, 30, np.max(spectrum), peak1+25, 20]
                elif attempt == 3:
                    p0 = [0.05, 30, np.max(spectrum), peak1+50, 20]
                #p0 = [0.05, 30, np.max(spectrum), 7473, 20]
                popt, _ = curve_fit(gaussianx, x_vals, spectrum, p0=p0)
                fit_curve = gaussianx(x_vals, *popt)
                attempt += 1              
            return pred_class, popt, fit_curve
        except RuntimeError:
            tqdm.write("Gaussian fit failed for one component")
            pred_class=0
            return pred_class, None, None
        
    elif pred_class == 2:
        tqdm.write(f"Predicted class: {pred_class} (Peak positions: {peak1}, {peak2})")
        # Fit two-component Gaussian: use initial guesses based on the known central wavelength 5303 Å.

            # Use the left and right peak positions as initial guesses
        print(amp1, amp2)
        p0 = [0.05,1,amp1, peak2, sigma1,
                  amp2, peak1, sigma2]
        try:
            popt, _ = curve_fit(double_gaussianx, x_vals, spectrum, p0=p0)
            fit_curve = double_gaussianx(x_vals, *popt)
            # Check if both peaks are on the same side of 5315
            tqdm.write(f"Fitted two-component Gaussian parameters: {popt}")
            attempt = 0
            max_attempts=3
            while ((popt[3] < s and popt[6] < s) or (popt[3] >= s and popt[6] >= s) or (np.abs(popt[4]) < 11) or (np.abs(popt[7]) < 11) or (popt[2]<0) or (popt[5]<0)) and attempt < max_attempts:
                if attempt == 0:
                    tqdm.write(f"Both peaks are on the same side of ({s}), ({popt[3]}, {popt[6]}) or less than sigma threshold of 14 ({popt[4],popt[7]}). Refitting with shifted peak.")
                    p0 = [0.05,5,np.max(spectrum), peak2-50, 26,
                    np.max(spectrum)/1.2, peak1-50, 18]
                    try:
                        popt, _ = curve_fit(double_gaussianx, x_vals, spectrum, p0=p0)
                        tqdm.write(f"Fitted two-component Gaussian parameters: {popt}")
                        pred_class=2
                        fit_curve = double_gaussianx(x_vals, *popt)
                    except RuntimeError:
                        popt=[0,0,0,0,0,0,0,0,0]
                        fit_curve = 0
                elif attempt == 1:
                    tqdm.write(f"Both peaks are on the same side of ({s}), ({popt[3]}, {popt[6]}). Refitting with closer peak.")
                    p0 = [0.05,5,np.max(spectrum), s-15, 20,
                    np.max(spectrum)/1.2, s+15, 22]
                    try:
                        popt, _ = curve_fit(double_gaussianx, x_vals, spectrum, p0=p0)
                        tqdm.write(f"Fitted two-component Gaussian parameters: {popt}")
                        pred_class=2
                        fit_curve = double_gaussianx(x_vals, *popt)
                    except RuntimeError:
                        popt=[0,0,0,0,0,0,0,0,0]
                        fit_curve = 0
                elif attempt == 2:
                    tqdm.write(f"Both peaks are on the same side of ({s}) ({popt[3]}, {popt[6]}). Refitting with single Gaussian peak.")
                # Refit with a single Gaussian
                    p0_single = [0.05, 4, np.max(spectrum), peak1, 20]
                    try:
                        popt, _ = curve_fit(gaussianx, x_vals, spectrum, p0=p0_single)
                    
                        tqdm.write(f"Refitted one-component Gaussian parameters: {popt}")
                        fit_curve = gaussianx(x_vals, *popt)
                        popt = np.append(popt, [100, s, 100])
                        attempta = 1
                        max_attemptas = 6  # Prevent infinite loop
                        pred_class=1
                        while (popt[4] < 20 or popt[2]<0) and attempta < max_attemptas:
                            tqdm.write(f"Peak is too narrow (sigma={popt[4]:.2f}). Re-fitting with shifted guess.")
                            if attempta == 1:
                                p0 = [0.05, 30, np.max(spectrum), peak2, 20]
                            else:
                                p0 = [0.05, 30, np.max(spectrum), (peak2 -(attempta)*25), 20]
                            popt, _ = curve_fit(gaussianx, x_vals, spectrum, p0=p0)

                            fit_curve = gaussianx(x_vals, *popt)
                            popt = np.append(popt, [100, s, 100])
                            attempta += 1  
                            pred_class=1
                    except RuntimeError:
                        popt=[0,0,0,0,0,0,0,0,0]
                        fit_curve = 0
                attempt += 1
            # Check if both peaks are valid (not too narrow and not negative)
            # If peaks are valid, return the two-Gaussian fit
            
            return pred_class, popt, fit_curve

        except RuntimeError:
            try:
                tqdm.write(f"Couldn't fit appropriate gaussian. Refitting with hardcoded peak position.")
                p0 = [0.05,1,np.max(spectrum), 5225, 15,
                0.2, 5365, 24]
                popt, _ = curve_fit(double_gaussianx, x_vals, spectrum, p0=p0)
                tqdm.write(f"Fitted two-component Gaussian parameters: {popt}")
                pred_class=2
                fit_curve = double_gaussianx(x_vals, *popt)
                if ((popt[3] < s and popt[6] < s) or (popt[3] >= s and popt[6] >= s) or (popt[4] < 11) or (popt[7] < 11) or (popt[2]<0) or (popt[5]<0)):
                    tqdm.write(f"Both peaks are on the same side of ({s}) ({popt[3]}, {popt[6]}). Refitting with single Gaussian peak.")
                # Refit with a single Gaussian
                    idx_max = np.argmax(spectrum)
                    mu_init = x_vals[idx_max]
                    p0_single = [0.05, 4, spectrum[idx_max], mu_init, 15]
                    try:
                        popt, _ = curve_fit(gaussianx, x_vals, spectrum, p0=p0_single)
                    
                        tqdm.write(f"Refitted one-component Gaussian parameters: {popt}")
                        fit_curve = gaussianx(x_vals, *popt)
                        popt = np.append(popt, [100, s, 100])
                        attempta = 1
                        max_attemptas = 6  # Prevent infinite loop
                        while (popt[4] < 12 or popt[2]<0) and attempta < max_attemptas:
                            tqdm.write(f"Peak is too narrow (sigma={popt[4]:.2f}). Re-fitting with shifted guess.")
                            if attempta == 1:
                                p0 = [0.05, 30, np.max(spectrum), peak2, 20]
                            else:
                                p0 = [0.05, 30, np.max(spectrum), (peak2 -(attempta)*25), 20]
                            popt, _ = curve_fit(gaussianx, x_vals, spectrum, p0=p0)

                            fit_curve = gaussianx(x_vals, *popt)
                            popt = np.append(popt, [100, s, 100])
                            attempta += 1  
                        pred_class=1
                    except RuntimeError:
                        popt=[0,0,0,0,0,0,0,0,0]
                        fit_curve = 0
                return pred_class, popt, fit_curve

            except RuntimeError:
                #replot = input(f" Replot one gaussian or skip?: ")
                #if replot=='y':
                try:
                    p0 = [0.05,10,np.max(spectrum), peak2, 25]
                    popt, _ = curve_fit(gaussianx, x_vals, spectrum, p0=p0)
                    pred_class=1
                    fit_curve = gaussianx(x_vals, *popt)
                    attempt = 0
                    max_attempts = 5  # Prevent infinite loop
                    while (popt[4] < 20 or popt[2]<0) and attempt < max_attempts:
                        tqdm.write(f"Peak is too narrow (sigma={popt[4]:.2f}). Re-fitting with shifted guess.")
                        if attempt == 0:
                            p0 = [0.05, 30, np.max(spectrum), peak1, 20]
                        else:
                            p0 = [0.05, 30, np.max(spectrum), (peak2 -(attempt)*25), 20]
                        popt, _ = curve_fit(gaussianx, x_vals, spectrum, p0=p0)
                        fit_curve = gaussianx(x_vals, *popt)
                        attempt += 1 
                except RuntimeError:
                    pred_class=0
                    popt=np.nan
            return pred_class, popt, fit_curve
    else:
        tqdm.write("No peak detected.")
        return pred_class, None, None




######################## MAIN EXECUTION FILE ###############################

def implement_inn(params_file, reg1,X,Y,y_pos, model, plot_spectrum=True, fit_plot=True):
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
    
    
    # Load the FITS file and extract the spectrum

    #reg1, data = sum_region(fits_file, x_coords, y_coords, sum=False)
    example_spectrum = reg1[min_pixel:max_pixel]

    

    if plot_spectrum:
        plt.plot(t[min_pixel:max_pixel], example_spectrum)
        plt.xlabel('Wavelength (Angstroms)')
        plt.ylabel('Flux')
        plt.title('Example Spectrum')
        plt.show()

    #n_samples = 10000
    #X_synthetic = []
    #y_synthetic = []
    #for i in range(n_samples):
    #    cls = np.random.choice([0, 1, 2])
    #    spec = simulate_spectrum(cls, min_pixel,max_pixel,lmin,lmax,l,A, A1, A2)
    #    # Standardize each spectrum individually
    #    spec_std = (spec - np.mean(spec)) / (np.std(spec) if np.std(spec) != 0 else 1)
    #    X_synthetic.append(spec_std)
    #    y_synthetic.append(cls)

    #X = np.array(X_synthetic)
    #Y = np.array(y_synthetic)
    



    x_vals = t[min_pixel:max_pixel]

    pred_class, popt, fit_curve = detect_and_fit(example_spectrum, model, x_vals,lambda_max,lambda_min,s,amp1,mu1,sigma1,amp2,mu2,sigma2)

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

