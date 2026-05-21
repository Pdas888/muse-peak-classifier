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
from scipy.optimize import minimize
import json
import pyfiglet
from termcolor import colored
import emcee
# -----------------------------
# 1. Helper Functions
# -----------------------------
# Builds a 1D pixel grid with similar configuration as MUSE WFM AO mode. Tweak this to suite your instrument. 
# 3722 is the total number of pixel in the spectrum
# 4699.95654296875 is the starting wavelength at pixel 1.
# 1.25 is the step
t = np.array([4699.95654296875 + i * 1.25 for i in range(1, 3722)])  
#t = np.array([4749.9814453125 + i * 1.25 for i in range(1, 3682)])  # MUSE WFM NOAO mode
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
        A = np.random.choice(np.arange(A-5, A+10, 0.01), size=1)[0]        # Randomized peak amplitude based on actual amplitude ranges.
        m = np.random.choice(np.arange(-0.05, 0.01, 0.01), size=1)[0]   # Randomized slope parameters to fit various continuum
        s = np.random.choice(np.arange(18, 40, 0.1), size=1)[0]         # Randomized sigma values to reflect varying velocity width in actual data. ******** Change to 12-30 for Fe XIV.
        mu = np.random.choice(np.arange(l-100, l+50, 0.1), size=1)[0]   # Randomized occurance of peak in the spectrum for robust classification
        spec += gaussian(t[indices], m, 3, A, mu=mu, sigma=s)
        spec += np.random.normal(0, n, size=spec.shape)                 # Adding noise to simulate real data
        peak_pos = [mu, -10.0]

    elif class_label == 2:
        A1 = np.random.choice(np.arange(A1-20, A1+15, 0.1), size=1)[0]    # Randomized ampltitude
        A2 = np.random.choice(np.arange(A2-15, A2+10, 0.1), size=1)[0]
        sigma1 = np.random.choice(np.arange(16, 30, 0.1), size=1)[0]    # Randomized sigma
        sigma2 = np.random.choice(np.arange(16, 30, 0.1), size=1)[0]
        m = np.random.choice(np.arange(-0.07, 0.07, 0.01), size=1)[0]
        mu1 = np.random.choice(np.arange(lmin, l-7, 0.1), size=1)[0]    # Randomized peak position. However, we employ a separation window of 15 A so that the two peaks may
        mu2 = np.random.choice(np.arange(l+8, lmax, 0.1), size=1)[0]    # blend but are always distinct - reflecting the real dataset. 
        spec += double_gaussian(t[indices], m, 5, A1, mu1, sigma1, A2, mu2, sigma2)
        spec += np.random.normal(0, n, size=spec.shape)
        peak_pos = [mu1, mu2]

    elif class_label == 3:
        # No broad peaks, a narrow peak and a flat continuum (e.g., a narrow peak at 5303 Å)
        A = np.random.choice(np.arange(A-2, A+25, 0.1), size=1, replace=True)
        m=np.random.choice(np.arange(-0.05, 0.01, 0.01), size=1, replace=True)
        s=np.random.choice(np.arange(3, 5, 0.1), size=1, replace=True)
        # One broad peak on, say, the left side of 5303 Å (e.g., around 5280 Å)
        mu=np.random.choice(np.arange(l-100, l+100, 1), size=1, replace=True)
        spec += gaussian(t[indices],m,3, A, mu=mu, sigma=s)
        spec += np.random.normal(0, 3, size=spec.shape)

    elif class_label == 0:
        noise_c = np.random.randint(1, 3)  # returns 1 or 2

        if noise_c == 1:
            m = np.random.choice(np.arange(-0.09, 0, 0.01), size=1)[0]
            spec += m * t[indices] + 5 + np.random.normal(0, n, size=spec.shape)

        if noise_c == 2:
            wavelengths = t[indices]  # use same wavelength samples as noise_c==1
            t_norm = (wavelengths - wavelengths.min()) / (wavelengths.max() - wavelengths.min())

            direction = np.random.choice([1, -1])
            continuum = 2.5 + 2.5 * (t_norm if direction == 1 else (1 - t_norm))**0.8

            absorption = np.zeros(len(wavelengths))
            rng = np.random.default_rng()  # no fixed seed so each call differs

            centers = rng.uniform(wavelengths.min(), wavelengths.max(), 100)
            depths  = rng.uniform(0.05, 0.5, 100)
            widths  = rng.uniform(1.5, 8.0, 100)

            for c, d, w in zip(centers, depths, widths):
                absorption -= d * np.exp(-0.5 * ((wavelengths - c) / w)**2)

            pixel_noise = np.random.normal(0, 0.5, len(wavelengths))
            spec = continuum + absorption + pixel_noise
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
# ─────────────────────────────────────────────────────────────────────────────
#  Noise estimator
# ─────────────────────────────────────────────────────────────────────────────
 
def _estimate_noise(spectrum, frac=0.10):
    """Estimate per-pixel noise from the line-free edges of the spectrum."""
    n = max(int(frac * len(spectrum)), 5)
    edge = np.concatenate([spectrum[:n], spectrum[-n:]])
    return max(np.std(edge), 1e-6)
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  Bayesian single-Gaussian fit
#  popt ordering: [m, c, A, mu, sigma]   identical to gaussianx args
# ─────────────────────────────────────────────────────────────────────────────
 
def _bayes_single(x_vals, spectrum, noise, mu_init):
    """
    Bayesian single-Gaussian + linear baseline fit.
 
    Priors — all soft (no hard bounds, walkers never get trapped):
      m       ~ Normal(0, 0.005)            slope stays near zero
      c       ~ Normal(median, 3*std)       baseline near spectrum level
      ln(A)   ~ Normal(ln(A_init), 1.0)    log-normal keeps A > 0 naturally
      mu      ~ Normal(mu_init, 40 A)      centroid near ML prediction
      ln(sig) ~ Normal(ln(25), 0.5)        log-normal centres sigma ~25 A
                                            2-sigma range covers ~9 to 68 A
 
    Strategy: Nelder-Mead MAP first → emcee walkers seeded from MAP.
    This makes convergence fast (< 500 steps) and robust.
    """
    med     = np.median(spectrum)
    std     = np.std(spectrum) + 1e-6
    idx0    = np.argmin(np.abs(x_vals - mu_init))
    A_init  = max(float(spectrum[idx0]) - med, 0.01)
 
    # ── Log-prior ────────────────────────────────────────────────────────────
    def log_prior(p):
        m, c, A, mu, sigma = p
        if A <= 0 or sigma <= 0:
            return -np.inf                   # only hard constraint: positivity
        lp  = -0.5 * (m / 0.005) ** 2
        lp += -0.5 * ((c - med) / (3 * std)) ** 2
        lp += -0.5 * ((np.log(A) - np.log(A_init)) / 1.0) ** 2
        lp += -0.5 * ((mu - mu_init) / 40.0) ** 2
        lp += -0.5 * ((np.log(sigma) - np.log(25.0)) / 0.5) ** 2
        return lp
 
    # ── Log-likelihood ───────────────────────────────────────────────────────
    def log_likelihood(p):
        return -0.5 * np.sum(((spectrum - gaussianx(x_vals, *p)) / noise) ** 2)
 
    def log_posterior(p):
        lp = log_prior(p)
        return lp + log_likelihood(p) if np.isfinite(lp) else -np.inf
 
    # ── MAP via Nelder-Mead ──────────────────────────────────────────────────
    p0 = np.array([0.0, med, A_init, mu_init, 25.0])
    res = minimize(lambda p: -log_posterior(p), p0, method='Nelder-Mead',
                   options={'maxiter': 3000, 'xatol': 1e-4, 'fatol': 1e-4})
    p_map = res.x
    p_map[2] = abs(p_map[2]) or A_init      # keep A > 0
    p_map[4] = abs(p_map[4]) or 25.0        # keep sigma > 0
 
    # ── emcee seeded from MAP ────────────────────────────────────────────────
    ndim, nw, nstep, nburn = 5, 16, 400, 80
    rng = np.random.default_rng(42)
    p0w = np.tile(p_map, (nw, 1))
    p0w[:, 0] += rng.normal(0, 1e-4, nw)                     # m
    p0w[:, 1] += rng.normal(0, 1e-2, nw)                     # c
    p0w[:, 2] *= np.exp(rng.normal(0, 0.05, nw))             # A  (log-space)
    p0w[:, 3] += rng.normal(0, 0.3,  nw)                     # mu
    p0w[:, 4] *= np.exp(rng.normal(0, 0.05, nw))             # sigma (log-space)
 
    sampler = emcee.EnsembleSampler(nw, ndim, log_posterior)
    sampler.run_mcmc(p0w, nstep, progress=False)
    flat = sampler.get_chain(discard=nburn, flat=True)
 
    popt = np.median(flat, axis=0)
    pstd = np.std(flat, axis=0)
    score = _posterior_score(flat, log_posterior)
    return popt, pstd, score          # popt: [m, c, A, mu, sigma]
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  Bayesian double-Gaussian fit
#  popt ordering: [m, c, A1, mu1, sig1, A2, mu2, sig2]  identical to double_gaussianx
# ─────────────────────────────────────────────────────────────────────────────
 
def _bayes_double(x_vals, spectrum, noise, mu1_init, mu2_init, s_ref):
    """
    Bayesian double-Gaussian + linear baseline fit.
 
    Key physical constraints:
      - mu1 < s_ref < mu2  HARD constraint (blue component left of rest, red right)
        This is the single most important prior — prevents collapse to one broad peak.
      - sigma prior: log-Normal centred on 30 A (FWHM ~71 A), width 0.25
        This is tight enough to prevent a single wide Gaussian absorbing both components.
      - mu priors centred on ML peaks with width 30 A (tighter than before).
    """
    mu1_init, mu2_init = np.sort([mu1_init, mu2_init])
 
    # If ML peaks are on the same side of s_ref, push them to sensible defaults
    if mu1_init >= s_ref:
        mu1_init = s_ref - 50.0
    if mu2_init <= s_ref:
        mu2_init = s_ref + 50.0
 
    med  = np.median(spectrum)
    std  = np.std(spectrum) + 1e-6
 
    def _amp(mu):
        idx = np.argmin(np.abs(x_vals - mu))
        lo, hi = max(0, idx - 5), min(len(spectrum), idx + 5)
        return max(float(np.mean(spectrum[lo:hi])) - med, 0.01)
 
    A1_init, A2_init = _amp(mu1_init), _amp(mu2_init)
 
    # ── Log-prior ────────────────────────────────────────────────────────────
    def log_prior(p):
        m, c, A1, mu1, sig1, A2, mu2, sig2 = p
        # Hard positivity constraints
        if A1 <= 0 or A2 <= 0 or sig1 <= 0 or sig2 <= 0:
            return -np.inf
        # HARD straddle constraint: blue face must be left of rest, red must be right
        # This is the key physical prior — prevents single-Gaussian collapse
        if mu1 >= s_ref or mu2 <= s_ref:
            return -np.inf
        lp  = -0.5 * (m / 0.005) ** 2
        lp += -0.5 * ((c - med) / (3 * std)) ** 2
        lp += -0.5 * ((np.log(A1) - np.log(A1_init)) / 1.0) ** 2
        lp += -0.5 * ((mu1 - mu1_init) / 30.0) ** 2          # tighter centroid prior
        lp += -0.5 * ((np.log(sig1) - np.log(30.0)) / 0.25) ** 2  # FWHM ~71 A, tight
        lp += -0.5 * ((np.log(A2) - np.log(A2_init)) / 1.0) ** 2
        lp += -0.5 * ((mu2 - mu2_init) / 30.0) ** 2
        lp += -0.5 * ((np.log(sig2) - np.log(30.0)) / 0.25) ** 2
        return lp
 
    # ── Log-likelihood ───────────────────────────────────────────────────────
    def log_likelihood(p):
        return -0.5 * np.sum(((spectrum - double_gaussianx(x_vals, *p)) / noise) ** 2)
 
    def log_posterior(p):
        lp = log_prior(p)
        return lp + log_likelihood(p) if np.isfinite(lp) else -np.inf
 
    # ── MAP via Nelder-Mead — start strictly on opposite sides of s_ref ──────
    p0 = np.array([0.0, med, A1_init, mu1_init, 30.0, A2_init, mu2_init, 30.0])
    res = minimize(lambda p: -log_posterior(p), p0, method='Nelder-Mead',
                   options={'maxiter': 8000, 'xatol': 1e-5, 'fatol': 1e-5})
    p_map = res.x
    # Safety: if MAP drifted to same side, reset to initialisations
    if p_map[3] >= s_ref or p_map[6] <= s_ref:
        tqdm.write("MAP drifted — resetting to ML initialisations.")
        p_map = p0.copy()
    for i in [2, 4, 5, 7]:
        p_map[i] = abs(p_map[i]) or 30.0
 
    # ── emcee seeded from MAP ────────────────────────────────────────────────
    ndim, nw, nstep, nburn = 8, 24, 400, 80
    rng = np.random.default_rng(42)
    p0w = np.tile(p_map, (nw, 1))
    p0w[:, 0] += rng.normal(0, 1e-4, nw)
    p0w[:, 1] += rng.normal(0, 1e-2, nw)
    p0w[:, 2] *= np.exp(rng.normal(0, 0.05, nw))
    p0w[:, 3] += rng.normal(0, 0.3,  nw)
    p0w[:, 4] *= np.exp(rng.normal(0, 0.05, nw))
    p0w[:, 5] *= np.exp(rng.normal(0, 0.05, nw))
    p0w[:, 6] += rng.normal(0, 0.3,  nw)
    p0w[:, 7] *= np.exp(rng.normal(0, 0.05, nw))
 
    sampler = emcee.EnsembleSampler(nw, ndim, log_posterior)
    sampler.run_mcmc(p0w, nstep, progress=False)
    flat = sampler.get_chain(discard=nburn, flat=True)
 
    # Correct any swapped components so mu1 < mu2 always
    swap = flat[:, 3] > flat[:, 6]
    flat[swap, 2], flat[swap, 5] = flat[swap, 5].copy(), flat[swap, 2].copy()
    flat[swap, 3], flat[swap, 6] = flat[swap, 6].copy(), flat[swap, 3].copy()
    flat[swap, 4], flat[swap, 7] = flat[swap, 7].copy(), flat[swap, 4].copy()
 
    popt = np.median(flat, axis=0)
    pstd = np.std(flat, axis=0)
    score = _posterior_score(flat, log_posterior)
    return popt, pstd, score          # popt: [m, c, A1, mu1, sig1, A2, mu2, sig2]
 
def _posterior_score(flat_samples, log_posterior_fn):
    """Mean log-posterior over accepted samples — used as per-spaxel Bayesian score."""
    scores = np.array([log_posterior_fn(p) for p in flat_samples[::10]])  # thin by 10 for speed
    return float(np.mean(scores[np.isfinite(scores)]))
# ─────────────────────────────────────────────────────────────────────────────
#  Main function — identical signature and return values to the original
# ─────────────────────────────────────────────────────────────────────────────
 
def detect_and_fit(spectrum, model, x_vals, lambda_max, lambda_min,
                   s=5315, mu_h1=5225, sigma1=15, mu_h2=5365, sigma2=15,
                   flux_err=None):
    """
    ML classification  →  Bayesian parameter estimation.
 
    Parameters
    ----------
    flux_err : float or None
        Per-pixel noise level. If None, estimated from edge pixels.
 
    Returns  (identical to original)
    -------
    pred_class : int          0, 1, or 2
    popt       : ndarray      posterior medians, same indexing as before
                   single → [m, c, A,  mu,  sigma]           popt[3] = centroid
                   double → [m, c, A1, mu1, sig1, A2, mu2, sig2]
                                                              popt[3] = blue centroid
                                                              popt[6] = red  centroid
    fit_curve  : ndarray      model evaluated at x_vals with popt
    """
 
    # ── Noise ────────────────────────────────────────────────────────────────
    noise = float(flux_err) if flux_err is not None else _estimate_noise(spectrum)
 
    # ── ML prediction (unchanged) ────────────────────────────────────────────
    spec_std = (spectrum - np.mean(spectrum)) / (np.std(spectrum) or 1.0)
    pred_probs, pred_peaks = model.predict(spec_std.reshape(1, -1))
    pred_class = int(np.argmax(pred_probs[0]))
    scaled_peaks = pred_peaks[0] * (lambda_max - lambda_min) + lambda_min
    peak1, peak2 = float(scaled_peaks[0]), float(scaled_peaks[1])
 
    tqdm.write(f"Predicted class: {pred_class}  |  ML peaks: {peak1:.1f}, {peak2:.1f}")
 
    # ─────────────────────────────────────────────────────────────────────────
    #  CLASS 1  —  single Gaussian
    # ─────────────────────────────────────────────────────────────────────────
    if pred_class == 1:
 
        def valid_single(p):
            if p is None: return False
            return p[2] > 0 and p[4] > 10.0 and x_vals.min() <= p[3] <= x_vals.max()
 
        trials = [peak1, peak1 - 25, peak1 + 25, peak1 - 50, peak1 + 50]
        popt = fit_curve = None
 
        for mu_try in trials:
            if not (x_vals.min() <= mu_try <= x_vals.max()):
                continue
            try:
                popt_try, pstd_try, score = _bayes_single(x_vals, spectrum, noise, mu_try)
                if valid_single(popt_try):
                    popt      = popt_try
                    fit_curve = gaussianx(x_vals, *popt)
                    tqdm.write(f"Single accepted  mu={popt[3]:.2f} ± {pstd_try[3]:.2f}  "
                               f"sigma={popt[4]:.2f} ± {pstd_try[4]:.2f}")
                    break
                tqdm.write(f"Single rejected at mu_init={mu_try:.1f}: {popt_try}")
            except Exception as e:
                tqdm.write(f"Single Bayes failed at mu_init={mu_try:.1f}: {e}")
 
        if popt is None:
            tqdm.write("All single attempts failed → class=0")
            return 0, None, None, None
 
        return pred_class, popt, fit_curve, score
 
    # ─────────────────────────────────────────────────────────────────────────
    #  CLASS 2  —  double Gaussian
    # ─────────────────────────────────────────────────────────────────────────
    elif pred_class == 2:
 
        peak1, peak2 = np.sort([peak1, peak2])
 
        def valid_double(p):
            if p is None or len(p) < 8: return False
            A1, mu1, sig1 = p[2], p[3], p[4]
            A2, mu2, sig2 = p[5], p[6], p[7]
            if not (A1 > 0 and A2 > 0 and sig1 > 10.0 and sig2 > 10.0):
                return False
            if not ((mu1 < s < mu2) or (mu2 < s < mu1)):
                return False
            return (x_vals.min() <= mu1 <= x_vals.max()
                    and x_vals.min() <= mu2 <= x_vals.max())
 
        # ML peaks first, then systematic shifts, hardcoded last
        trials = [
            (peak1,      peak2),
            (peak1 - 25, peak2),
            (peak1,      peak2 - 25),
            (peak1 - 25, peak2 - 25),
            (peak1 + 25, peak2),
            (peak1,      peak2 + 25),
            (peak1 - 50, peak2),
            (peak1,      peak2 - 50),
            (mu_h1,      mu_h2),
        ]
 
        popt = fit_curve = None
 
        for mu1_try, mu2_try in trials:
            if not (x_vals.min() <= mu1_try <= x_vals.max()
                    and x_vals.min() <= mu2_try <= x_vals.max()):
                tqdm.write(f"Skipping out-of-range: {mu1_try:.1f}, {mu2_try:.1f}")
                continue
            try:
                popt_try, pstd_try, score = _bayes_double(
                    x_vals, spectrum, noise, mu1_try, mu2_try, s_ref=s)
                if valid_double(popt_try):
                    popt      = popt_try
                    fit_curve = double_gaussianx(x_vals, *popt)
                    tqdm.write(f"Double accepted  "
                               f"mu1={popt[3]:.2f} ± {pstd_try[3]:.2f}  "
                               f"mu2={popt[6]:.2f} ± {pstd_try[6]:.2f}")
                    break
                tqdm.write(f"Double rejected at ({mu1_try:.1f}, {mu2_try:.1f}): {popt_try}")
            except Exception as e:
                tqdm.write(f"Double Bayes failed at ({mu1_try:.1f}, {mu2_try:.1f}): {e}")
 
        # Single-Gaussian fallback
        if popt is None:
            tqdm.write("All double attempts failed → single fallback.")
            try:
                popt_s, _, score = _bayes_single(x_vals, spectrum, noise, peak1)
                if popt_s[2] > 0 and popt_s[4] > 1.0:
                    popt       = np.concatenate([popt_s, [np.nan, np.nan, np.nan]])
                    fit_curve  = gaussianx(x_vals, *popt_s)
                    pred_class = 1
                    tqdm.write(f"Single fallback accepted as class=1")
            except Exception as e:
                tqdm.write(f"Single fallback failed: {e}")
 
        if popt is None:
            tqdm.write("All attempts failed → NaN popt.")
            return 0, np.full(8, np.nan), None, None
 
        return pred_class, popt, fit_curve, score
 
    # ─────────────────────────────────────────────────────────────────────────
    #  CLASS 0  —  no peak
    # ─────────────────────────────────────────────────────────────────────────
    else:
        tqdm.write("No peak detected.")
        return pred_class, None, None, None
 

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
        plt.xlabel('Wavelength (Angstroms)', fontsize=16)
        plt.ylabel('Flux', fontsize=16)
        plt.title('Example Spectrum', fontsize=16)
        plt.show()

    x_vals = t[min_pixel:max_pixel]

    pred_class, popt, fit_curve, score = detect_and_fit(example_spectrum, model, x_vals,lambda_max,lambda_min,s,mu1,sigma1,mu2,sigma2)

    # Plot the original spectrum and the fitted Gaussian(s) if available.
    if fit_plot:
        plt.figure(figsize=(10, 5))
        plt.plot(x_vals, example_spectrum, label='Original Spectrum')
        if fit_curve is not None:
            plt.plot(x_vals, fit_curve, label='Gaussian Fit', linestyle='--')
        plt.xlabel('Wavelength (Å)', fontsize=16)
        plt.ylabel('Flux', fontsize=16)
        plt.legend(fontsize=12)
        plt.title(f"Detection Result: Class {pred_class}", fontsize=12)
        plt.show()
    return pred_class, popt, fit_curve, score

