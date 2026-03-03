import numpy as np
import matplotlib.pyplot as plt

blueshift=np.load('C:/Users/z5391280/OneDrive - UNSW/Desktop/PhD/MUSE/my_reduction/ML peak detection/Peak classifier/blueshift00_S.npy')
redshift=np.load('C:/Users/z5391280/OneDrive - UNSW/Desktop/PhD/MUSE/my_reduction/ML peak detection/Peak classifier/redshift00_S.npy')
blue_brightness=np.load('C:/Users/z5391280/OneDrive - UNSW/Desktop/PhD/MUSE/my_reduction/ML peak detection/Peak classifier/blueshifted00_S_sb.npy')
red_brightness=np.load('C:/Users/z5391280/OneDrive - UNSW/Desktop/PhD/MUSE/my_reduction/ML peak detection/Peak classifier/redshift00_S_sb.npy')
blue_width=np.load('C:/Users/z5391280/OneDrive - UNSW/Desktop/PhD/MUSE/my_reduction/ML peak detection/Peak classifier/blueshifted00_S_width.npy')
red_width=np.load('C:/Users/z5391280/OneDrive - UNSW/Desktop/PhD/MUSE/my_reduction/ML peak detection/Peak classifier/redshift00_S_width.npy')

print(blueshift.shape)
plt.imshow(red_brightness, origin='lower', norm='log', vmin=10, vmax=4000)