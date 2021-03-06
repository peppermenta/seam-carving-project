# Worked on by Mansi Nanavati

from PIL import Image
from tqdm import trange
import numpy as np
from energy_maps import gradient_energy_map
from generate_dino_map import dino_energy_map
from energy_maps import MajorBlobMap
import numba
import matplotlib.pyplot as plt


energy_maps = [MajorBlobMap, dino_energy_map]

def crop_c(img, scale_c):
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
  # print(r,c)
  new_c = int(scale_c * c)

  for i in trange(c - new_c):
    img = carve_column(img)

  return img

def crop_r(img, scale_r):
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
  img = crop_c(img, scale_r)
  img = np.rot90(img, 3, (0, 1))
  return img

@numba.jit
def carve_column(img):
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

  M, backtrack = minimum_seam(img)
  mask = np.ones((r, c), dtype=np.bool)

  j = np.argmin(M[-1])
  for i in reversed(range(r)):
    mask[i, j] = False
    j = backtrack[i, j]

  mask = np.stack([mask] * 3, axis=2)
  img = img[mask].reshape((r, c - 1, 3))
  return img

@numba.jit
def minimum_seam(img):
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
  energy_map = np.zeros((r,c))
  temp = np.zeros((r,c))

  # Combining energy maps
  for func in energy_maps:
    temp = func(img)
    energy_map = np.maximum(np.divide(temp, np.linalg.norm(temp)), energy_map)
  
  energy_map += gradient_energy_map(img)

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

img = Image.open('input/castle.jpg')
# img = Image.open('input/eiffel.jpg')
# img = Image.open('input/sunset2.jpg')
# img = Image.open('input/boating.jpg')
# img = Image.open('input/monkey.jpg')
# img = Image.open('input/bike.jpg')
# img = Image.open('input/beach.jpg')
# img = Image.open('input/dog.jpg')
img = np.array(img)

resize1 = crop_c(img, 0.8)
vanilla_resize = Image.fromarray(resize1)
vanilla_resize.save('output/castle_best_map.png')
fig = plt.figure(figsize=(10, 7))
rows = 1
cols = 2
fig.add_subplot(rows, cols, 1)
plt.imshow(img)
fig.add_subplot(rows, cols, 2)
plt.imshow(resize1)
plt.show()