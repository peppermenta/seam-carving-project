import numpy as np
import sys
import os
from PIL import Image
from tqdm import trange
from scipy.ndimage.filters import convolve
import numba
import matplotlib.pyplot as plt

from energy_maps import MajorBlobMap, gradient_energy_map, saliency_energy_map
from seam_carving import *

def areaRatio(sal_map, carved_sal_map): 
    # extract one channel
    sal_map = sal_map[:, :, 0]
    carved_sal_map = carved_sal_map[:, :, 0]

    sal_area = np.count_nonzero(sal_map)
    carved_sal_area = np.count_nonzero(carved_sal_map)

    ratio = (carved_sal_area / sal_area)
    return ratio

def evaluateEnergyMap(folder, energyMap, num_images=1000):
    i = 0
    sum = 0

    file1 = open("filenames.txt", "a")  # append mode
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            file1.write(filename)
            file1.write("\n")

            
            print(filename)
            if i >= num_images:
                break
            i = i + 1

            img_path = os.path.join(folder, filename)
            # img_path = os.path.join(folder, "75.jpg")
            sal_path = img_path[:-3] + "png"

            img = Image.open(img_path)
            img = np.array(img)
            sal_img = Image.open(sal_path)
            sal_img = np.array(sal_img)
            # copying to three channels
            sal_img = np.stack([sal_img] * 3, axis=2)
            carved_img, carved_sal = crop_c_fast_eval(img, sal_img, 0.8, energyMap)

            # print("ratio:", areaRatio(sal_img, carved_sal))
            sum = sum + areaRatio(sal_img, carved_sal)
    
    file1.close()

    avg = sum / num_images
    return avg


folder = r"C:\Users\soumi\Documents\IIT-Hyderabad\Assignmets, quizzes & tests\IVP\MSRA10K_Imgs_GT\MSRA10K_Imgs_GT\Imgs"

MajorBlobMap_avg = evaluateEnergyMap(folder, MajorBlobMap)
# gradient_energy_map_avg = evaluateEnergyMap(folder, gradient_energy_map)
# saliency_energy_map_avg = evaluateEnergyMap(folder, saliency_energy_map)

print("Major Blob Map Area Retention:", MajorBlobMap_avg)
# print("Gradient Map Area Retention:", gradient_energy_map_avg)
# print("Saliency Map Area Retention:", saliency_energy_map_avg)



# img_path = folder + r"\75.jpg"
# sal_path = folder + r"\75.png"
# img = Image.open(img_path)
# img = np.array(img)
# sal_img = Image.open(sal_path)
# sal_img = np.array(sal_img)
# # copying to three channels
# sal_img = np.stack([sal_img] * 3, axis=2)
# carved_img, carved_sal = crop_c_fast_eval(img, sal_img, 0.8, MajorBlobMap)

# # print(sal_img[:, :, 0].shape)
# # print(carved_sal.shape)

# print("ratio:", areaRatio(sal_img, carved_sal))
# # temp = carved_sal[:, :, 0] - carved_sal[:, :, 1]
# # print(np.count_nonzero(temp))


# fig = plt.figure(figsize=(10, 7))
# rows = 1
# cols = 2
# fig.add_subplot(rows, cols, 1)
# plt.imshow(sal_img)
# fig.add_subplot(rows, cols, 2)
# plt.imshow(carved_sal)
# plt.show()

