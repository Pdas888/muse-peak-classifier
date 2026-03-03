"""
This script contains the main functions to train a multitask neural network for spectral peak detection and parameter regression, and to apply this model 
to a FITS datacube to extract doppler shifts, surface brightness, and velocity widths for emission lines. The script includes the following key components:
1) neural_model_multitask: Defines and trains a multitask neural network that performs both classification (number of peaks) and regression (peak positions).
2) batch_process: A helper function that processes a batch of pixel coordinates, applies the trained model to the spectra at those coordinates,
 and computes the desired parameters while handling validity checks and logging discarded results.
3) fit_curve: The main function that loads the FITS datacube, prepares the data, trains the model, and applies it across the specified spatial 
region using parallel processing for efficiency.

*****Make change in the Main section to control the execution of the code. Set plot_all to True to visualize the spectrum and fit for each pixel,
 and set load_file to True to load existing results from .npy files instead of processing the FITS data.*****

Make sure to adjust file paths, parameters, and model architecture as needed for your specific use case.
Author: [Priyam Das]
Date: [3/03/2026]
"""

import numpy as np
from astropy.io import fits
from tqdm import tqdm
import time
#from astropy.wcs import WCS
from concurrent.futures import ProcessPoolExecutor, as_completed
from detect_peak_v2 import implement_inn, simulate_spectrum

import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report,confusion_matrix, accuracy_score
#import tensorflow as tf
#from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# Wavelength grid construction
# Starting wavelength = 4699.9565 Å
# Spectral sampling = 1.25 Å per pixel
# Total spectral bins = 3721
t = np.array([4699.95654296875 + i * 1.25 for i in range(1, 3722)])
# Log file to record discarded spectra/pixels during preprocessing
discard_log = open("C:/Users/z5391280/OneDrive - UNSW/Desktop/PhD/MUSE/my_reduction/ML peak detection/Peak classifier/logs/discarded_pixels_S00.log", "a")


def neural_model_multitask(X_train, y_train_cls, y_train_pos, X_test, y_test_cls, y_test_pos, lambda_max,lambda_min):
    """
    Multitask neural network for spectral peak analysis.

    Tasks:
    1) Classification:
       Predict number of emission peaks in a spectrum:
           0 → no peak
           1 → single peak
           2 → double peak

    2) Regression:
       Predict peak centroid positions (max 2 peaks).
       Positions are normalized to [0, 1] during training.

    Parameters
    ----------
    X_train : array (N_train, N_lambda)
        Input spectra for training.
    y_train_cls : array (N_train,)
        Integer class labels (0, 1, 2).
    y_train_pos : array (N_train, 2)
        Normalized peak positions in [0,1].
    X_test, y_test_cls, y_test_pos : test equivalents.
    lambda_max, lambda_min : float
        Physical wavelength bounds used for normalization.

    Returns
    -------
    model : trained Keras model
    """

    # ==============================
    # Model Architecture
    # ==============================

    # Input layer: one node per wavelength bin

    inp = Input(shape=(X_train.shape[1],))
    x = Dense(128, activation='relu')(inp)
    x = Dropout(0.1)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)

    out_class = Dense(3, activation='softmax', name='classification')(x)
    out_pos = Dense(2, activation='sigmoid', name='peak_positions')(x)

    model = Model(inputs=inp, outputs=[out_class, out_pos])
    model.compile(optimizer='adam',
                  loss={'classification': 'sparse_categorical_crossentropy',
                        'peak_positions': 'mae'},
                  loss_weights={'classification': 1.0, 'peak_positions': 1.5},
                  metrics={'classification': 'accuracy'})
    
    # ==============================
    # Training
    # ==============================


    model.fit(X_train, {'classification': y_train_cls, 'peak_positions': y_train_pos},
              epochs=50, batch_size=32, validation_split=0.15, verbose=0)
    
    # Predict on test set
    pred_probs_test, _ = model.predict(X_test)

    # Convert probabilities to class labels
    y_pred_class = np.argmax(pred_probs_test, axis=1)
    acc = accuracy_score(y_test_cls, y_pred_class)
    f1  = f1_score(y_test_cls, y_pred_class, average='macro')



    cm = confusion_matrix(y_test_cls, y_pred_class, labels=[0,1,2])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6,5))

    im = ax.imshow(cm_norm, interpolation='nearest', cmap='viridis')

    ax.set_xticks([0,1,2])
    ax.set_yticks([0,1,2])
    ax.set_xticklabels(['No peak', '1 peak', '2 peaks'])
    ax.set_yticklabels(['No peak', '1 peak', '2 peaks'])

    ax.set_xlabel('Predicted class')
    ax.set_ylabel('True class')
    ax.set_title('Normalized confusion matrix')

    # Annotate values
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm_norm[i,j]:.2f}",
                    ha="center", va="center",
                    color="white" if cm_norm[i,j] > 0.5 else "cyan", fontsize=12)

    plt.colorbar(im, ax=ax, label='Fraction')
    plt.tight_layout()
    plt.show()
    print(classification_report(
    y_test_cls,
    y_pred_class,
    target_names=['No peak', '1 peak', '2 peaks'],
    digits=3
    ))


    # Predict on test set
    pred_probs_test, pred_pos_test = model.predict(X_test)
    y_pred_class = np.argmax(pred_probs_test, axis=1)

    # Rescale predicted and true peak positions
    #global lambda_min, lambda_max
    #lambda_min = 5200
    #lambda_max = 5450
    
    #lambda_min = 7478
    #lambda_max = 7670
    pred_pos_phys = pred_pos_test * (lambda_max - lambda_min) + lambda_min
    true_pos_phys = y_test_pos * (lambda_max - lambda_min) + lambda_min

    mask_1 = (y_test_cls == 1)

    # Use only the first peak for class 1
    err_1 = pred_pos_phys[mask_1, 0] - true_pos_phys[mask_1, 0]

    mae_1 = np.mean(np.abs(err_1))
    rmse_1 = np.sqrt(np.mean(err_1**2))

    print(f"Class 1 (single peak): MAE = {mae_1:.2f} Å, RMSE = {rmse_1:.2f} Å")
    mask_2 = (y_test_cls == 2)

    pred_sorted = np.sort(pred_pos_phys[mask_2], axis=1)
    true_sorted = np.sort(true_pos_phys[mask_2], axis=1)

    err_2 = pred_sorted - true_sorted

    mae_2 = np.mean(np.abs(err_2))
    rmse_2 = np.sqrt(np.mean(err_2**2))

    print(f"Class 2 (two peaks): MAE = {mae_2:.2f} Å, RMSE = {rmse_2:.2f} Å")
    mask_1_correct = (y_test_cls == 1) & (y_pred_class == 1)
    mask_2_correct = (y_test_cls == 2) & (y_pred_class == 2)

    # Class 1
    err_1c = pred_pos_phys[mask_1_correct, 0] - true_pos_phys[mask_1_correct, 0]
    mae_1c = np.mean(np.abs(err_1c))
    rmse_1c = np.sqrt(np.mean(err_1c**2))

    # Class 2
    pred_sorted_c = np.sort(pred_pos_phys[mask_2_correct], axis=1)
    true_sorted_c = np.sort(true_pos_phys[mask_2_correct], axis=1)
    err_2c = pred_sorted_c - true_sorted_c
    mae_2c = np.mean(np.abs(err_2c))
    rmse_2c = np.sqrt(np.mean(err_2c**2))

    print(f"Class 1 (correct only): MAE={mae_1c:.2f}, RMSE={rmse_1c:.2f}")
    print(f"Class 2 (correct only): MAE={mae_2c:.2f}, RMSE={rmse_2c:.2f}")
    plt.hist(err_1c, bins=30, alpha=0.7, label='1 peak')
    plt.hist(err_2c.flatten(), bins=30, alpha=0.7, label='2 peaks')
    plt.xlabel(r'$\Delta \lambda$ (Å)')
    plt.ylabel('Count')
    plt.legend()
    plt.show()



    return model


def batch_process(pixels_batch, clone, params_file, model, l, plot_spectrum, fit_plot):
    """Process a batch of (i, j) pixel coordinates. 
    Computes doppler shift, surface brightness and velocity width from the prediction and parametrs achieved by implementing fit_curve
    on the spectrum at each pixel coordinate i,j. 
    Parameters
    ----------
    pixels_batch: list
      list of pixels in batch
    clone: array 
    3D numpy array of the datacube containing spectral data
    params_file: .json file containing the parameters for the simulated spectra and the rest wavelength
    model: trained Keras model for peak detection and parameter regression
    l: rest wavelength of the emission line being analysed
    plot_spectrum: boolean flag to control whether to plot the spectrum and fit for each pixel
    fit_plot: boolean flag to control whether to plot the fit results for each pixel

    Returns
    -------
    results: list of tuples containing (i, j, v1, v2, sb1, sb2, vw1, vw2) for each pixel in the batch
    where v1,v2 are the doppler shifts for the blueshifted and redshifted peaks respectively (nan if not present or invalid),
    sb1, sb2 are the surface brightness for the blueshifted and redshifted peaks respectively (nan if not present or invalid),
    and vw1, vw2 are the velocity widths for the blueshifted and redshifted peaks respectively (nan if not present or invalid).
    """
    results = []

    for i, j in pixels_batch:
        try:
            spec = np.average(clone[:, j-2:j+2, i-2:i+2], axis=(1,2))        # average over 4x4 spatial pixels to improve S/N  
            pred, popt, _ = implement_inn(params_file, spec, model, plot_spectrum, fit_plot)

            if pred == 1:
                v = (299792.458) * (l - popt[3]) / (l)  # Doppler shift for single peak case
                sb=(popt[2]*popt[4]*25)/(0.3989)        # Surface brightness for single peak case (using Gaussian area formula)
                vw= (2.35482*(popt[4]*299792.458))/(popt[3])    # Velocity width for single peak case (FWHM converted to velocity units)
                if v < 9000 and v > 0:
                    results.append((i, j, v, np.nan,sb,np.nan,vw, np.nan))
                elif v <0 and v > -9000:
                    results.append((i, j, np.nan, v,np.nan,sb,np.nan,vw))
                else:
                    discard_log.write(f"Discarded (pred==1) at ({i},{j}): v={v:.2f}, surface brightness={sb:.2f}, width={vw:.2f}\n")

            elif pred == 2:
                v1 = (299792.458) * (l - popt[3]) / (l) # Doppler shift for first peak
                v2 = (299792.458) * (l - popt[6]) / (l) # Doppler shift for second peak
                sb1=(popt[2]*popt[4]*25)/(0.3989)         # Surface brightness for first peak (using Gaussian area formula)
                sb2=(popt[5]*popt[7]*25)/(0.3989)       # Surface brightness for second peak (using Gaussian area formula)
                vw1=(2.35482*(popt[4]*299792.458))/(popt[3])    # Velocity width for first peak (FWHM converted to velocity units)
                vw2=(2.35482*(popt[7]*299792.458))/(popt[3])    # Velocity width for second peak (FWHM converted to velocity units)
                # define validity flags
                is_v1_valid   = (0 < v1 <= 9000)
                is_v2_valid   = (-9000 <= v2 < 0)
                is_swapped    = (-9000 <= v1 < 0) and (0 < v2 <= 9000)

                if is_v1_valid and is_v2_valid:
                    # both in the “correct” ranges
                    results.append((i, j, v1, v2, sb1,sb2,vw1,vw2))

                elif is_swapped:
                    # v1/v2 have the right sign but in opposite roles → swap them
                    results.append((i, j, v2, v1,sb2,sb1,vw2,vw1))

                elif is_v2_valid:
                    # only v2 is trustworthy
                    results.append((i, j, np.nan, v2,np.nan,sb2,np.nan,vw2))

                elif is_v1_valid:
                    # only v1 is trustworthy
                    results.append((i, j, v1, np.nan,sb1,np.nan,vw1, np.nan))

                else:
                    # neither is in any acceptable window
                    discard_log.write(
                        f"Discarded (pred==2) at ({i},{j}): "
                        f"v1={v1:.2f}, v2={v2:.2f}\n"
                        f"surface brightness={sb1:.2f}, {sb2:.2f}"
                        f"width velocity = {vw1:.2f}, {vw2:.2f}"
                    )
            else:
                results.append((i, j, np.nan, np.nan, np.nan,np.nan,np.nan,np.nan))
                discard_log.write(f"Discarded (pred==0) at ({i},{j}): pred={pred}\n")

        except Exception as e:
            tqdm.write(f"Failed at pixel ({i},{j}): {e}")
            results.append((i, j, np.nan, np.nan, np.nan,np.nan,np.nan,np.nan))
            discard_log.write(f"Failed at pixel ({i},{j}): {e}\n")

    return results



def fit_curve(fits_file, params_file, x_start, x_final, y_start, y_final, plot_all=False,load_file=False):
    """
    Main function to fit curves to spectra extracted from a FITS datacube.
    Reads the necessary parameters from the .json file, prepares training data for the model, trains the model, 
    and processes the specified pixel region in batches to compute doppler shifts, surface brightness and velocity widths.

    Parameters
    ----------
    fits_file : str
        Path to the FITS file containing the datacube.
    params_file : str
        Path to the .json file containing parameters for simulated spectra and rest wavelength.
    x_start, x_final, y_start, y_final : int
        Pixel coordinates defining the region to process.
    plot_all : bool
        Flag to control whether to plot the spectrum and fit for each pixel.
    load_file : bool
        Flag to control whether to load existing results from .npy files instead of processing the FITS data.
    
    Returns
    -------
    blueshift, redshift, sb_b, sb_r, vw_b, vw_r : 2D numpy arrays
        Arrays containing the blueshift, redshift, surface brightness and velocity width for the blueshifted and redshifted peaks respectively.
        Values are nan for pixels where the corresponding peak is not present or the prediction is invalid.
    
    """
    img3 = fits.open(fits_file)
    result = img3[0].data
    img3.close()
    clone = result
    rows, cols = clone.shape[1], clone.shape[2]
    if load_file:
        blueshift=np.load('C:/Users/z5391280/OneDrive - UNSW/Desktop/PhD/MUSE/my_reduction/ML peak detection/Peak classifier/blueshift00_S.npy')
        redshift=np.load('C:/Users/z5391280/OneDrive - UNSW/Desktop/PhD/MUSE/my_reduction/ML peak detection/Peak classifier/redshift00_S.npy')
        blue_brightness=np.load('C:/Users/z5391280/OneDrive - UNSW/Desktop/PhD/MUSE/my_reduction/ML peak detection/Peak classifier/blueshifted00_S_sb.npy')
        red_brightness=np.load('C:/Users/z5391280/OneDrive - UNSW/Desktop/PhD/MUSE/my_reduction/ML peak detection/Peak classifier/redshift00_S_sb.npy')
        blue_width=np.load('C:/Users/z5391280/OneDrive - UNSW/Desktop/PhD/MUSE/my_reduction/ML peak detection/Peak classifier/blueshifted00_S_width.npy')
        red_width=np.load('C:/Users/z5391280/OneDrive - UNSW/Desktop/PhD/MUSE/my_reduction/ML peak detection/Peak classifier/redshift00_S_width.npy')
    else:
        blueshift = np.full((rows, cols), np.nan, dtype=float)
        redshift = np.full((rows, cols), np.nan, dtype=float)
        blue_brightness= np.full((rows, cols), np.nan, dtype=float)
        red_brightness= np.full((rows, cols), np.nan, dtype=float)
        blue_width= np.full((rows, cols), np.nan, dtype=float)
        red_width= np.full((rows, cols), np.nan, dtype=float)

    with open(params_file, 'r') as file:
            data = json.load(file)
    min_pixel = data["Simulated_spectra_parameters"]["xmin"]
    max_pixel = data["Simulated_spectra_parameters"]["xmax"]
    lmin = data["Simulated_spectra_parameters"]["lambda_min"]
    lmax = data["Simulated_spectra_parameters"]["lambda_max"]
    l = data["Simulated_spectra_parameters"]["lambda"]
    A= data["Simulated_spectra_parameters"]["Amplitude"]
    A1= data["Simulated_spectra_parameters"]["Amplitude1"]
    A2= data["Simulated_spectra_parameters"]["Amplitude2"]
    n=data["Simulated_spectra_parameters"]["noise"]
    lambdao=data["Rest_wavelength"]

    n_samples = 1000               # Total number of synthetic spectra to generate for training
    X_synthetic = []
    y_synthetic = []
    y_pos=[]
    for i in range(n_samples):
        cls = np.random.choice([0, 1, 2])
        spec, peaks = simulate_spectrum(cls, min_pixel,max_pixel,lmin,lmax,l,A, A1, A2,n)
        # Standardize each spectrum individually
        spec_std = (spec - np.mean(spec)) / (np.std(spec) if np.std(spec) != 0 else 1)
        X_synthetic.append(spec_std)
        y_synthetic.append(cls)
        y_pos.append(peaks)

    X = np.array(X_synthetic)
    Y = np.array(y_synthetic)
    y_pos = np.array(y_pos)
    y_pos_norm = (y_pos - lmin) / (lmax - lmin)

    X_train, X_test, y_train_class, y_test_class, y_train_pos, y_test_pos = train_test_split(
    X, Y, y_pos_norm,
    test_size=0.2, 
    random_state=42,
    stratify=Y                  # only use stratify on classification labels
    )

    model = neural_model_multitask(X_train, y_train_class, y_train_pos, X_test, y_test_class, y_test_pos,lmax,lmin) #train the model and evaluate on test set
    while True:
        user_input = input("Are you satisfied with the training? (y/n): ").strip().lower()

        if user_input == 'y':
            print("Proceeding with current model.")
            break

        elif user_input == 'n':
            print("Retraining model...")
            model = neural_model_multitask(
                X_train, y_train_class, y_train_pos,
                X_test, y_test_class, y_test_pos,
                lmax, lmin
            )

        else:
            print("Invalid input. Please enter 'y' or 'n'.")




    pixels = [(i, j) for j in range(y_start, y_final) for i in range(x_start, x_final)] #generate list of pixel coordinates to process
    # Vectorized batch processing with ProcessPoolExecutor for faster execution. We will divide the pixels into batches and process each batch in parallel.
    # The batch_process function will handle the processing of each batch and return the results, which we will then aggregate into the final blueshift and redshift arrays.

    batch_size = 50  # Adjust this based on your system's capabilities
    pixel_batches = [pixels[i:i + batch_size] for i in range(0, len(pixels), batch_size)]

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(batch_process, batch, clone, params_file, model, lambdao,plot_all,plot_all) for batch in pixel_batches] #submit each batch to the executor for parallel processing

        # tqdm to track how many batches are done
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Pixels 🚀",dynamic_ncols=True, leave=True):
            batch_results = future.result()
            for i, j, blue_val, red_val, blue_sb, red_sb, blue_vw, red_vw in batch_results:
                blueshift[j, i] = blue_val
                redshift[j, i] = red_val
                blue_brightness[j,i]=blue_sb
                red_brightness[j,i]= red_sb
                blue_width[j, i]=blue_vw
                red_width[j, i]=red_vw
    return blueshift, redshift,blue_brightness, red_brightness, blue_width, red_width

if __name__ == "__main__":
    start_time = time.time()
    fits_file = "C:/Users/z5391280/OneDrive - UNSW/Desktop/PhD/MUSE/FITS_files_0509-67.5/gauss_smooth.fits"
    params_file = "C:/Users/z5391280/OneDrive - UNSW/Desktop/PhD/MUSE/my_reduction/ML peak detection/Peak classifier/params_sulphur.json"

    #blueshift, redshift= fit_curve(fits_file,params_file, 147,162,94,104)
    #blueshift, redshift = fit_curve(fits_file, params_file,  210,250,90,250)
    blueshift, redshift, sb_b, sb_r, vw_b, vw_r = fit_curve(fits_file, params_file, 109, 111, 140,142, plot_all=True, load_file=True)

    #blueshift, redshift = fit_curve(fits_file, params_file,  120,124,168,172)
    end_time = time.time()     # stop timing
    print(f"\n Completed in {end_time - start_time:.2f} seconds!")

    np.save('C:/Users/z5391280/OneDrive - UNSW/Desktop/PhD/MUSE/my_reduction/ML peak detection/Peak classifier/blueshift00_S.npy', blueshift)
    np.save('C:/Users/z5391280/OneDrive - UNSW/Desktop/PhD/MUSE/my_reduction/ML peak detection/Peak classifier/redshift00_S.npy', redshift)
    np.save('C:/Users/z5391280/OneDrive - UNSW/Desktop/PhD/MUSE/my_reduction/ML peak detection/Peak classifier/blueshifted00_S_sb.npy', sb_b)
    np.save('C:/Users/z5391280/OneDrive - UNSW/Desktop/PhD/MUSE/my_reduction/ML peak detection/Peak classifier/redshift00_S_sb.npy', sb_r)
    np.save('C:/Users/z5391280/OneDrive - UNSW/Desktop/PhD/MUSE/my_reduction/ML peak detection/Peak classifier/blueshifted00_S_width.npy', vw_b)
    np.save('C:/Users/z5391280/OneDrive - UNSW/Desktop/PhD/MUSE/my_reduction/ML peak detection/Peak classifier/redshift00_S_width.npy', vw_r)

