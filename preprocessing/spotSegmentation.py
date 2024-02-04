"""
Documentation for the given code:

1. Import necessary libraries/modules:
    - numpy as np: Library for numerical operations.
    - cv2: OpenCV library for image processing.
    - sys, os, math: Modules for system-specific parameters and functions, operating system functions, and mathematical operations.
    - segment_anything: Custom module for segmentation tasks.
"""
import numpy as np
import cv2, sys, os, math
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

"""
2. Define constants and variables:
    - HOME: Current working directory.
    - CHECKPOINT_PATH: Path to the pre-trained model checkpoint.
    - model_type: Type of the pre-trained model.
    - device: Device to use for model computations (e.g., "cuda").
"""

HOME = os.getcwd()
sys.path.append("..")

sam_checkpoint = CHECKPOINT_PATH = os.path.join(HOME, "weights\\sam_vit_h_4b8939.pth")
model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

"""
3. Define a function GetSilhouette(img) to extract the silhouette from the input image:
    - Read the input image and convert it to RGB format.
    - Generate a mask using the SamAutomaticMaskGenerator.
    - Process the generated masks and extract the silhouette.
    - Handle special cases and defects in the silhouette.
    - Return the final silhouette image.
"""

#Get healthy silhouette 
def GetSilhouette(img):
  try:
    image_bgr = cv2.imread(img)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    mask_generator = SamAutomaticMaskGenerator(model=sam,
      points_per_side=64,
      pred_iou_thresh=0.69,
      stability_score_thresh=0.6,
      #stability_score_offset=0.5,
      crop_n_layers=1,
      crop_n_points_downscale_factor=1,
      min_mask_region_area=25, )
    
    sam_result = mask_generator.generate(image_rgb)

    masks = [
        mask
        for mask
        in sorted(sam_result, key=lambda x: x['area'], reverse=True)
    ]

    img_shape = image_bgr.shape
    obj_mask = masks[0]
    temp_mask = obj_mask["segmentation"]
    temp_mask = np.asarray(temp_mask*255, dtype="uint8")
    init_mask = None

    # sam_mask["bbox"]  -> x [0]        y [1]       width [2]                  height [3]
    # img.shape         -> height [0]   width [1]   channels [2]

    half_height = int(image_bgr.shape[0]/2)
    half_width = int(image_bgr.shape[1]/2)

    # All Background, no fruit
    if obj_mask["bbox"][2] == (img_shape[1]-1) and obj_mask["bbox"][3] == (img_shape[0]-1):
      init_mask = temp_mask
      init_mask[init_mask==255] = 150
      init_mask[init_mask==0] = 255
      init_mask[init_mask==150] = 0

    # One of the sides of the BBox is equal to the size of the length/width of the image
    elif obj_mask["bbox"][2] == (img_shape[1]-1) or obj_mask["bbox"][3] == (img_shape[0]-1):
      if temp_mask[half_height, half_width] == 255:
        init_mask = temp_mask
      else:
        for i in range(1, len(masks)):
          temp_mask = masks[i]["segmentation"]
          temp_mask = np.asarray(temp_mask*255, dtype="uint8")
          if temp_mask[half_height, half_width] == 255:
            init_mask = temp_mask
            break

    # Background without fill with vertical/horizontal lines, no fruit
    elif (obj_mask["bbox"][0] == ((math.ceil(img_shape[1] - obj_mask["bbox"][2])/2) - 1)) or (obj_mask["bbox"][1] == (math.ceil((img_shape[0] - obj_mask["bbox"][3])/2) - 1)):
      for i in range(1, len(masks)):
        temp_mask = masks[i]["segmentation"]
        temp_mask = np.asarray(temp_mask*255, dtype="uint8")
        if temp_mask[half_height, half_width] == 255:
          init_mask = temp_mask
          break

    # Normal case
    else:
      init_mask = np.asarray(obj_mask["segmentation"]*255, dtype="uint8")

    # Special cases
    if init_mask is None:
      init_mask = np.asarray(obj_mask["segmentation"]*255, dtype="uint8")

    # Searching for defects
    intersection = []
    #Add first fruit silhouette (entire) 
    intersection.append(init_mask)

    for i in range(len(masks)):
      _mask = masks[i]["segmentation"]
      _mask = np.asarray(_mask*255, dtype="uint8")

      bool_mask = np.asarray(_mask, dtype="bool")
      bool_init_mask = np.asarray(init_mask, dtype="bool")
      bool_result = np.logical_and(bool_init_mask, bool_mask)

      if np.sum(bool_result) > 0 and (np.sum(bool_mask) <= int(np.sum(bool_init_mask) * 0.75)):
        intersection.append(_mask)

    result = intersection[0]
    c = np.zeros(result.shape,dtype="uint8")
    if len(intersection) > 0:
      for i in range(1, len(intersection)):
        #Spot defect restricting areas less than 450
        area = masks[i]["area"]
        if area <= 450:
          c = cv2.add(c,intersection[i])
      
    kernel = np.ones((3,3), np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=2)

    return c
  except:
    err = open("fruitSilhouetteError.txt","a")
    err.write(img+"\n")
    return None

#Execute segmentation process for all healthy images

rgbImagesPath = "images/spot/rgb"
nirImagesPath = "images/spot/nir"

rgbImagesLabels = os.listdir(rgbImagesPath)
nirImageLabels = os.listdir(nirImagesPath)
counter = 1

"""
4. Iterate over the labels of RGB images:
    - Segment each RGB image to obtain the silhouette.
    - Generate the corresponding NIR image label.
    - Segment the NIR image to obtain the silhouette.
    - Save the segmented RGB and NIR silhouettes to the specified directories.
"""

for label in rgbImagesLabels:
  print("Segmentando imagen:",label)

  fileName = "newSil_" + str(counter).zfill(4) + ".png"

  #Generate rgb segmented image
  resultRgb = GetSilhouette(rgbImagesPath + "/" + label)

  #Generate nir segmented image
  nirLabel = label.replace("rgb","nir")
  resultNir = GetSilhouette(nirImagesPath + "/" + nirLabel)
  
  if resultNir != None and resultRgb != None:
    cv2.imwrite(os.path.join(HOME,"segmentation\\apple\\rgb\\" + fileName),resultRgb)
    cv2.imwrite(os.path.join(HOME,"segmentation\\apple\\nir\\" + fileName),resultNir) 
    counter += 1
