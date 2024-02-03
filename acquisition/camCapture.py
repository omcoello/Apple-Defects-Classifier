"""

Note: Detailed comments within the code provide additional information about specific functionalities and camera configurations.
Documentation for the given code:

1. Import necessary libraries/modules:
    - pylon: Interface to Basler's pylon Camera Software Suite.
    - genicam: Provides access to the GenICam GenApi interface.
    - gxipy: GigE Vision SDK for Python provided by Daheng Imaging (GxIAP).
    - PIL: Python Imaging Library for image processing.
    - numpy: Library for numerical computing.
    - datetime: Module for manipulating dates and times.
    - cv2: OpenCV library for computer vision tasks.
    - glob: Module for pathname pattern expansion.
    - tkinter: Standard GUI toolkit for Python.
    - os: Module providing a portable way of using operating system-dependent functionality.
"""

# Librerias para la camara
from pypylon import pylon
from pypylon import genicam
import gxipy as gx
from PIL import Image
import numpy
import datetime

# Librerias varias
import cv2
from glob import glob
import tkinter as tk
from tkinter import filedialog
import os

""""
2. Define configuration for active cameras:
    - RGB_active, NIR_active, THERMAL_active: Boolean variables indicating whether respective cameras are active.
"""

# Configuración de cámaras activas
RGB_active = True
NIR_active = True
THERMAL_active = False

"""
3. Define function: configure_camera(cam)
    - Opens the camera device.
    - Retrieves and prints device information.
    - Sets camera parameters such as width, height, acquisition mode, etc.
    - Configures image format conversion.
    - Starts grabbing images.
    - Returns an image format converter object.
"""
def configure_camera(cam):
	
	cam.Open()
	print("DeviceClass: ", cam.GetDeviceInfo().GetDeviceClass())
	print("DeviceFactory: ", cam.GetDeviceInfo().GetDeviceFactory())
	print("ModelName: ", cam.GetDeviceInfo().GetModelName())
	print("Cam IP: ", cam.GetDeviceInfo().GetIpAddress())
	
	new_width = cam.Width.GetValue() - cam.Width.GetInc()
	if new_width >= cam.Width.GetMin():
		cam.Width.SetValue(new_width)
	
	cam.MaxNumBuffer = 5
	cam.AcquisitionMode.SetValue('Continuous')
	cam.Width = 1280
	cam.Height = 1024
	cam.CenterX.SetValue(True)
	cam.CenterY.SetValue(True)
	converter = pylon.ImageFormatConverter()
	converter.OutputPixelFormat = pylon.PixelType_BGR8packed
	converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
	cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
	return converter


"""
4. Define function: get_saving_folder()
    - Opens a Tkinter dialog to select a folder for saving images.
    - Checks if required subfolders ('rgb', 'nir') exist within the selected folder.
    - If any subfolder is missing, displays an error message and prompts for folder selection again.
    - Returns the selected folder path.
"""
def get_saving_folder():
    root = tk.Tk()
    root.withdraw()

    folder = filedialog.askdirectory()

    folders = ["rgb", "nir"]
    missing_folders = []

    for folder_name in folders:
        folder_path = os.path.join(folder, folder_name)
        if not os.path.exists(folder_path):
            missing_folders.append(folder_name)

    if missing_folders:
        missing_folders_str = ", ".join(missing_folders)
        tk.messagebox.showerror("Error", f"Las carpetas {missing_folders_str} no existen en la ruta seleccionada.")
        return get_saving_folder()

    return folder


"""
5. Define function: captureImages(path)
    - Captures and saves images from active cameras based on the provided path.
    - Images are saved with a specific naming convention based on the camera type and a sequential number.
"""
def captureImages(path):

	print("save images")
	aux=path+"/rgb/*.png"
	number = len(glob(aux)) + 1
	str_number = str(number).zfill(4)
	number += 1 if number != 0 else None
	if RGB_active:
		cv2.imwrite(path+f"/rgb/rgb_{str_number}.png", pimg)		
	if NIR_active:
		cv2.imwrite(path+f"/nir/nir_{str_number}.png", img_nir)
	if THERMAL_active:
		cv2.imwrite(path+f"/thermal/thermal_{str_number}.png", img_thermal)
	print(f"Captured: {number}")

"""
6. Initialize saving folder path using get_saving_folder() function.
"""
ruta=get_saving_folder()


"""
7. Try-except block for camera initialization and image capturing:
    - Enumerates connected cameras.
    - Configures and initializes active cameras.
    - Enters a loop for continuous image capturing and processing until a termination key is pressed.
    - Handles grabbing images from active cameras.
    - Improves image quality (if applicable).
    - Displays images using OpenCV.
    - Allows capturing images manually by pressing 'c' or 'space' keys.
    - Terminates the loop and releases camera resources on 'ESC' key press or program completion.
"""
try:
	devices = pylon.TlFactory.GetInstance().EnumerateDevices()	   

	if RGB_active:
		device_manager = gx.DeviceManager()
		dev_num, dev_info_list = device_manager.update_device_list()
		cam = device_manager.open_device_by_index(1)
		cam.TriggerMode.set(gx.GxSwitchEntry.OFF)		
		cam.BalanceWhiteAuto.set(2)
		cam.Gain.set(18)

		if cam.GammaParam.is_readable():
			gamma_value = cam.GammaParam.get()
			gamma_lut = gx.Utility.get_gamma_lut(gamma_value)
		else:
			gamma_lut = None
			
		if cam.ContrastParam.is_readable():
			contrast_value = cam.ContrastParam.get()
			contrast_lut = gx.Utility.get_contrast_lut(contrast_value)
		else:
			contrast_lut = None
		if cam.ColorCorrectionParam.is_readable():
			color_correction_param = cam.ColorCorrectionParam.get()
		else:
			color_correction_param = 0

		cam.data_stream[0].set_acquisition_buffer_number(1)
	
	if NIR_active:
		nircam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(devices[0]))
		print("\nConfiguring NIR Camera:")
		nir_converter = configure_camera(nircam)

	if THERMAL_active:
		thermal_cap = cv2.VideoCapture(0)

	keypress = 0
	initalTime = datetime.datetime.now()
	imageCounter = 0
	captureBool = False
	while (NIR_active and nircam.IsGrabbing()):
		timeActual = datetime.datetime.now() - initalTime
		actual = timeActual.microseconds
		timeActual = timeActual.seconds
		
		if keypress == 48:
			captureBool = True
			initialTime = datetime.datetime.now()
		if keypress == 49:
			captureBool = False
			

		if timeActual >= 1.9 and captureBool: 
			initalTime = datetime.datetime.now()
			print("Tiempo total transcurrido:",initalTime.second)
			
			print("Imagen numero:",imageCounter)
			imageCounter +=1
			captureImages(ruta)
		
		if imageCounter == 20:
			imageCounter = 0
			initialTime = datetime.datetime.now()
			captureBool = False
		
				
		if RGB_active:	
			# start data acquisition
			cam.stream_on()
			
			# acquisition image: num is the image number
			# get raw image
			raw_image = cam.data_stream[0].get_image()
			if raw_image is None:
				print("Getting image failed.")
				continue

			# get RGB image from raw image
			rgb_image = raw_image.convert("RGB")
			if rgb_image is None:
				continue

			# improve image quality
			rgb_image.image_improvement(color_correction_param, contrast_lut, gamma_lut)

			# create numpy array with data from raw image
			numpy_image = rgb_image.get_numpy_array()
			if numpy_image is None:
				continue

			#display image with opencv
			pimg = cv2.cvtColor(numpy.asarray(numpy_image),cv2.COLOR_BGR2RGB)
			resized = cv2.resize(pimg, (640,512), interpolation = cv2.INTER_CUBIC)
			cv2.line(resized,(0,256),(140,256),(255,0,0),8)
			cv2.line(resized,(320,0),(320,65),(255,0,0),8)
			cv2.imshow("VISIBLE-RGB",resized)

			# print height, width, and frame ID of the acquisition image
			#print("Frame ID: %d   Height: %d   Width: %d" % (raw_image.get_frame_id(), raw_image.get_height(), raw_image.get_width()))

			# stop data acquisition
			cam.stream_off()			

		if NIR_active:
			nirGrabResult = nircam.RetrieveResult(500, pylon.TimeoutHandling_ThrowException)
			if nirGrabResult.GrabSucceeded():
				image = nir_converter.Convert(nirGrabResult)
				img_nir = image.GetArray()
				# Seleccionar solo el canal rojo (índice 0)
				img_nir = img_nir[:,:,0]
				#print("Resolucion NIR:",img_nir.shape)
				cv2.namedWindow('NIR', cv2.WINDOW_NORMAL)
				cv2.resizeWindow('NIR', 640, 512)
				cv2.imshow("NIR", img_nir)			

		if THERMAL_active:
			ret, img_thermal = thermal_cap.read()
			#print("Resolucion THERMAL:",img_thermal.shape)
			cv2.namedWindow('THERMAL', cv2.WINDOW_NORMAL)
			cv2.resizeWindow('THERMAL', 640, 480)
			cv2.imshow('THERMAL', img_thermal)

		if keypress == 27:
			break
		elif keypress in [99, 32]:  # 'c' or 'space'
			captureImages(ruta)

		keypress = cv2.waitKey(1)

	if RGB_active:
		# close device
		cam.close_device()
	if NIR_active:
		nircam.Close()
	cv2.destroyAllWindows()

except genicam.GenericException as e:
	print(f"ERROR04: An exception occurred. {e}")
	print(e.GetDescription())
