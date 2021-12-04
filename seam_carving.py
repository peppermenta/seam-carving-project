import sys

from PIL import Image
from tqdm import trange
import numpy as np
from scipy.ndimage.filters import convolve
import numba
import matplotlib.pyplot as plt
from energy_maps import MajorBlobMap

def crop_c(img, scale_c, energy_map_fn):
  '''
  Scale down image by cropping columns
  Code modified from https://karthikkaranth.me/blog/implementing-seam-carving-with-python/

  Parameters
  --------------------------------
  img: np.ndarray
    The image to resize
  scale_c: float
    Target scale to resize image
  energy_map_fn: function
    Function that takes an img as input and returns the energy map

  Returns
  -------------------------------
  out: np.ndarray
    Resized image
  '''
  r, c, _ = img.shape
  new_c = int(scale_c * c)

  for i in trange(c - new_c):
    img = carve_column(img, energy_map_fn)

  return img

def crop_r(img, scale_r, energy_map_fn):
  '''
  Scale down image by cropping rows
  Code modified from https://karthikkaranth.me/blog/implementing-seam-carving-with-python/

  Parameters
  --------------------------------
  img: np.ndarray
    The image to resize
  scale_r: float
    Target scale to resize image
  energy_map_fn: function
    Function that takes an img as input and returns the energy map

  Returns
  -------------------------------
  out: np.ndarray
    Resized image
  '''
  img = np.rot90(img, 1, (0, 1))
  img = crop_c(img, scale_r, energy_map_fn)
  img = np.rot90(img, 3, (0, 1))
  return img

@numba.jit
def carve_column(img, energy_map_fn):
  '''
  Remove a column from an image using seam carving
  Code modified from https://karthikkaranth.me/blog/implementing-seam-carving-with-python/

  Parameters
  --------------------------------
  img: np.ndarray
    The image to resize
  energy_map_fn: function
    Function that takes an img as input and returns the energy map

  Returns
  -------------------------------
  out: np.ndarray
    Image with a column removed
  '''
  r, c, _ = img.shape

  M, backtrack = minimum_seam(img, energy_map_fn)
  mask = np.ones((r, c), dtype=np.bool)

  j = np.argmin(M[-1])
  for i in reversed(range(r)):
    mask[i, j] = False
    j = backtrack[i, j]

  mask = np.stack([mask] * 3, axis=2)
  img = img[mask].reshape((r, c - 1, 3))
  return img

@numba.jit
def minimum_seam(img, energy_map_fn):
  '''
  Find the minimum seam for seam carving using dynamic programming
  Code modified from https://karthikkaranth.me/blog/implementing-seam-carving-with-python/

  Parameters
  --------------------------------
  img: np.ndarray
    The image to apply minimum seam finding on
  energy_map_fn: function
    Function that takes an img as input and returns the energy map

  Returns
  -------------------------------
  M: np.ndarray
    Energy map computed for the image

  backtrack: np.ndarray
    Minimum seam backtracking, same size as M
  '''
  r, c, _ = img.shape
  energy_map = energy_map_fn(img)

  M = energy_map.copy()
  backtrack = np.zeros_like(M, dtype=np.int)

  for i in range(1, r):
    for j in range(0, c):
      # Handle the left edge of the image, to ensure we don't index a -1
      if j == 0:
        idx = np.argmin(M[i-1, j:j + 2])
        backtrack[i, j] = idx + j
        min_energy = M[i-1, idx + j]
      else:
        idx = np.argmin(M[i - 1, j - 1:j + 2])
        backtrack[i, j] = idx + j - 1
        min_energy = M[i - 1, idx + j - 1]

      M[i, j] += min_energy

  return M, backtrack


@numba.jit
def minimum_seam_fast(img, energy_map):
  r, c, _ = img.shape
  # energy_map = energy_map_fn(img)

  M = energy_map.copy()
  backtrack = np.zeros_like(M, dtype=np.int)

  for i in range(1, r):
    for j in range(0, c):
      # Handle the left edge of the image, to ensure we don't index a -1
      if j == 0:
        idx = np.argmin(M[i-1, j:j + 2])
        backtrack[i, j] = idx + j
        min_energy = M[i-1, idx + j]
      else:
        idx = np.argmin(M[i - 1, j - 1:j + 2])
        backtrack[i, j] = idx + j - 1
        min_energy = M[i - 1, idx + j - 1]

      M[i, j] += min_energy

  return M, backtrack

@numba.jit
def carve_column_fast(img, energy_map):
  '''
  Remove a column from an image using seam carving
  Code modified from https://karthikkaranth.me/blog/implementing-seam-carving-with-python/

  Parameters
  --------------------------------
  img: np.ndarray
    The image to resize
  energy_map_fn: function
    Function that takes an img as input and returns the energy map

  Returns
  -------------------------------
  out: np.ndarray
    Image with a column removed
  '''
  r, c, _ = img.shape

  M, backtrack = minimum_seam_fast(img, energy_map)
  mask = np.ones((r, c), dtype=np.bool)

  j = np.argmin(M[-1])
  for i in reversed(range(r)):
    mask[i, j] = False
    j = backtrack[i, j]
  energy_mask = np.copy(mask)
  mask = np.stack([mask] * 3, axis=2)
  img = img[mask].reshape((r, c - 1, 3))
  energy_map = energy_map[energy_mask].reshape((r,c-1))
  return img, energy_map

def crop_c_fast(img, scale_c, energy_map_fn):
  '''
  Scale down image by cropping columns
  Code modified from https://karthikkaranth.me/blog/implementing-seam-carving-with-python/

  Parameters
  --------------------------------
  img: np.ndarray
    The image to resize
  scale_c: float
    Target scale to resize image
  energy_map_fn: function
    Function that takes an img as input and returns the energy map

  Returns
  -------------------------------
  out: np.ndarray
    Resized image
  '''
  energy_map = energy_map_fn(img)
  r, c, _ = img.shape
  new_c = int(scale_c * c)

  for i in trange(c - new_c):
    img, energy_map = carve_column_fast(img, energy_map)

  return img

def crop_r_fast(img, scale_r, energy_map_fn):
  '''
  Scale down image by cropping rows
  Code modified from https://karthikkaranth.me/blog/implementing-seam-carving-with-python/

  Parameters
  --------------------------------
  img: np.ndarray
    The image to resize
  scale_r: float
    Target scale to resize image
  energy_map_fn: function
    Function that takes an img as input and returns the energy map

  Returns
  -------------------------------
  out: np.ndarray
    Resized image
  '''
  img = np.rot90(img, 1, (0, 1))
  img = crop_c_fast(img, scale_r, energy_map_fn)
  img = np.rot90(img, 3, (0, 1))
  return img

def crop_c_fast_eval(img, saliency_img, scale_c, energy_map_fn):
  # print("shape:", img.shape, saliency_img.shape)
  '''
  Scale down image by cropping columns
  Code modified from https://karthikkaranth.me/blog/implementing-seam-carving-with-python/

  Parameters
  --------------------------------
  img: np.ndarray
    The image to resize
  scale_c: float
    Target scale to resize image
  energy_map_fn: function
    Function that takes an img as input and returns the energy map

  Returns
  -------------------------------
  out: np.ndarray
    Resized image
  '''
  energy_map = energy_map_fn(img)
  r, c, _ = img.shape
  new_c = int(scale_c * c)

  for i in trange(c - new_c):
    saliency_img, temp = carve_column_fast(saliency_img, energy_map)
    img, energy_map = carve_column_fast(img, energy_map)

  return img, saliency_img

def crop_r_fast_eval(img, saliency_img, scale_r, energy_map_fn):
  '''
  Scale down image by cropping rows
  Code modified from https://karthikkaranth.me/blog/implementing-seam-carving-with-python/

  Parameters
  --------------------------------
  img: np.ndarray
    The image to resize
  scale_r: float
    Target scale to resize image
  energy_map_fn: function
    Function that takes an img as input and returns the energy map

  Returns
  -------------------------------
  out: np.ndarray
    Resized image
  '''
  img = np.rot90(img, 1, (0, 1))
  saliency_img = np.rot90(saliency_img, 1, (0, 1))
  img, saliency_img = crop_c_fast_eval(img, saliency_img, scale_r, energy_map_fn)
  img = np.rot90(img, 3, (0, 1))
  saliency_img = np.rot90(saliency_img, 3, (0, 1))    

  return img, saliency_img


# img = Image.open('major.jpeg')
# img = np.array(img)
# # resize1 = crop_c(img, 0.8, MajorBlobMap)
# resize2 = crop_c_fast(img, 0.8, MajorBlobMap)
# # resize1 = crop_r_fast(img, 0.9, MajorBlobMap)
# # resize2 = crop_r(img, 0.9, MajorBlobMap)
# fig = plt.figure(figsize=(10, 7))
# rows = 1
# cols = 2
# fig.add_subplot(rows, cols, 1)
# plt.imshow(img)
# fig.add_subplot(rows, cols, 2)
# plt.imshow(resize2)
# # fig.add_subplot(rows, cols, 2)
# # plt.imshow(resize2)
# plt.show()
# # vanilla_resize = Image.fromarray(resize1)
# # vanilla_resize.save('vanilla.png')
# # major_resize = Image.fromarray(resize2)
# # major_resize.save('majorResize.png')
