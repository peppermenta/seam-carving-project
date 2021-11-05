import numpy as np
from PIL import Image
from scipy.ndimage.filters import convolve
import matplotlib.pyplot as plt

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

def convert_to_grayscale(img):
  gray = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
  for i in range(gray.shape[0]):
    for j in range(gray.shape[1]):
      gray[i][j] = int(gray[i][j])
  return gray

def computeHistogram(img):
    # initializing frequency array for intensities with all zeros
    histogram = np.zeros(256)
    # iterating over all pixels in image
    for row in img:
        for pixel in row:
            # updating frequency count of observed intensity
            p = int(pixel)
            histogram[p] = histogram[p] + 1
    # displaying image histogram
    x = np.linspace(0, 255, 256)
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.bar(x, histogram)
    plt.title('Image histogram')
    plt.show()
    # computing mode of image
    mode = -1
    count = 0
    # iterating through intensity frequency array for max intensity
    for i in range(256):
        if histogram[i] > count:
            count = histogram[i]
            mode = i
    modality = 0
    for i in range(254):
        if histogram[i + 1] > histogram[i] and histogram[i + 1] > histogram[i + 2]:
            modality += 1
    if histogram[0] > histogram[1]:
        modality += 1
    if histogram[255] > histogram[254]:
        modality += 1
    # print("Mode of histogram: " + str(mode))
    # print("Number of occurences: " + str(count))
    # print("Modality of histogram: " + str(modality))
    # returning computed mode and histogram of the image
    return histogram

def fasterOtsu(histogram, img):
    # calculating pmf for each pixel intensity
    p = [h / img.shape[0] ** 2 for h in histogram]
    # calculating cdf for each pixel intensity in accordance to the pmf
    P = [0 for i in range(256)]
    term = 0
    for i in range(256):
        term += p[i]
        P[i] = term
    mu_0 = np.zeros(256)
    mu_1 = np.zeros(256)
    sigma_2b = np.zeros(256)
    mu = np.mean(img)
    # sentinel values for best variance and intensity so far
    best = 0
    threshold = -1
    for t in range(255):
        # avoiding divide by zero errors
        if P[t + 1] == 0 or P[t + 1] == 1:
            continue
        # calculating new sigma value for current iteration
        mu_0[t + 1] = (mu_0[t] * P[t] + (t + 1) * p[t + 1]) / P[t + 1]
        mu_1[t + 1] = (mu - mu_0[t + 1] * P[t + 1]) / (1 - P[t + 1])
        sigma_2b[t + 1] = P[t + 1] * \
            (1 - P[t + 1]) * ((mu_0[t + 1] - mu_1[t + 1]) ** 2)
        # testing current iteration against best result so far
        if sigma_2b[t + 1] > best:
            best = sigma_2b[t + 1]
            threshold = t + 1
    # returning computed optimal threshold
    # print(sigma_2b)
    return threshold + 1

def binarize(img, threshold):
    # initializing binarized version to all zeros
    binaryImg = np.zeros(img.shape)
    # iterating over all pixels in image and applying thresholding activation
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] >= threshold:
                binaryImg[i][j] = 255
    # displaying binarized image
    plt.imshow(binaryImg, cmap='gray')
    return binaryImg

img = Image.open('major.jpeg')
img = np.array(img)
print(img.shape)
plt.imshow(img)
plt.show()
img = convert_to_grayscale(img)
plt.imshow(img, cmap='gray')
plt.show()
print(img.shape)
histogram = computeHistogram(img)
threshold = fasterOtsu(histogram, img)
print(threshold)
binary_img = binarize(img, threshold)
plt.imshow(binary_img, cmap='gray')
plt.show()