import numpy as np
from scipy.ndimage.filters import convolve

def gradient_energy_map(img):
  '''
  Basic energy map using gradient magnitudes
  Code taken from https://karthikkaranth.me/blog/implementing-seam-carving-with-python/

  Author: Tarun Ram (AI19BTECH11004)

  Parameters
  ----------------------------
  img : np.ndarray
    The image to compute energy map for 

  Returns
  ----------------------------
  energy_map: np.ndarray
    The computed energy map
  '''
  filter_du = np.array([
    [1.0, 2.0, 1.0],
    [0.0, 0.0, 0.0],
    [-1.0, -2.0, -1.0],
  ])
  # This converts it from a 2D filter to a 3D filter, replicating the same
  # filter for each channel: R, G, B
  filter_du = np.stack([filter_du] * 3, axis=2)

  filter_dv = np.array([
      [1.0, 0.0, -1.0],
      [2.0, 0.0, -2.0],
      [1.0, 0.0, -1.0],
  ])
  # This converts it from a 2D filter to a 3D filter, replicating the same
  # filter for each channel: R, G, B
  filter_dv = np.stack([filter_dv] * 3, axis=2)

  img = img.astype('float32')
  convolved = np.absolute(convolve(img, filter_du)) + np.absolute(convolve(img, filter_dv))

  # We sum the energies in the red, green, and blue channels
  energy_map = convolved.sum(axis=2)

  return energy_map