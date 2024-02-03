"""

Note: Detailed comments within the code provide additional information about specific functionalities and file operations.
Documentation for the given code:

1. Import necessary libraries/modules:
    - cv2: OpenCV library for computer vision tasks.
    - os: Module providing a portable way of using operating system-dependent functionality.
    - numpy as np: Library for numerical computing.
    - random: Module for generating random numbers and selecting random elements.
"""
import cv2 
import os
import numpy as np
import random

"""
2. Define file paths for original and destination directories:
    - rgbPath, nirPath: Paths to original RGB and NIR image directories.
    - rgbDestinyPath, nirDestinyPath: Paths to destination directories for augmented images.
    - rgbPath_sil, nirPath_sil: Paths to original RGB and NIR silhouette image directories.
    - rgbDestinyPath_sil, nirDestinyPath_sil: Paths to destination directories for augmented silhouette images.
    - fileListPath: Path to the file containing the list of randomly selected files.
"""

rgbPath = ""   
nirPath = ""
rgbDestinyPath = ""
nirDestinyPath = ""

rgbPath_sil = ""   
nirPath_sil = ""
rgbDestinyPath_sil = ""
nirDestinyPath_sil = ""
fileListPath="RamdonAugmentationBruisergb.txt"
total_images = 5000

rgbFiles = os.listdir(rgbPath)
nirFiles = os.listdir(nirPath)


"""
3. Calculate the number of augmented images needed:
    - num_original: Number of original files in the NIR directory.
    - augmented: Number of additional images required to reach the total_images target.
"""
#num_original = len(rgbFiles)
num_original= len(nirFiles)
augmented = total_images - num_original
print(f"Original files: {num_original}") 
print(f"Augmented needed: {augmented}")

#random_files = random.sample(rgbFiles, augmented)
random_files = random.sample(nirFiles, augmented)

counter = 0
num_original+=1


"""
4. Write selected filenames to the file specified by fileListPath.
"""

with open(fileListPath, "w") as file:
    for random_file in random_files:
        file.write(random_file + "/n")

"""
5. Loop through each randomly selected file for augmentation:
    - Read NIR images and corresponding silhouette images.
    - Perform image augmentation by flipping horizontally.
    - Save augmented images to the destination directories with sequential filenames.
"""

for i, file in enumerate(random_files):
    
    #rgbImg = cv2.imread(rgbPath + file)
    #rgbImg_sil = cv2.imread(rgbPath_sil + file)   
       
    nirImg = cv2.imread(nirPath + file)
    nirImg_sil = cv2.imread(nirPath_sil + file)
    
    if counter < augmented:
        # Aplicar una transformación de volteo aleatorio (horizontal o vertical)
        # flip_code = np.random.choice([-1, 0, 1])
        # rgbAugmented = cv2.flip(rgbImg, flip_code)
        # nirAugmented = cv2.flip(nirImg, flip_code)

        #rgbAugmented = cv2.flip(rgbImg,1)    
        #rgbAugmented_sil = cv2.flip(rgbImg_sil,1)    
        nirAugmented = cv2.flip(nirImg,1)
        nirAugmented_sil = cv2.flip(nirImg_sil,1)


        #cv2.imwrite(rgbDestinyPath + "rgbBruise" + str(num_original).zfill(4) + ".png",rgbAugmented)
        #cv2.imwrite(rgbDestinyPath_sil + "rgbBruise" + str(num_original).zfill(4) + ".png",rgbAugmented_sil)

        cv2.imwrite(nirDestinyPath + "nirBruise" + str(num_original).zfill(4) + ".png",nirAugmented)        
        cv2.imwrite(nirDestinyPath_sil + "nirBruise" + str(num_original).zfill(4) + ".png",nirAugmented_sil)       

        num_original+=1
        counter += 1
        
    if counter == augmented:
        break
        
print("Augmentation completed!")
