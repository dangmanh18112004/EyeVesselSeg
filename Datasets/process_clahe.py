import cv2
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def clahe_equalized(image):
	""" Note: This function used to process an image using 
		Contrast Limited Adaptive Histogram (CLAHE) """
	lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
	clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))

	# Apply clahe into Lightness of the image in LAB mode
	lab[...,0] = clahe.apply(lab[...,0])
	
	bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
	rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
	return rgb

def processFIVES(
		origin_folder = './Data/FIVES/Train',
		processed_folder = './Processed_data/FIVES'):
	
	""" Note: This function used to process all images in FIVES dataset folder """
	
	image_dir_path = origin_folder + '/Images/'
	mask_dir_path = origin_folder + '/Labels/'

	image_path_list = os.listdir(image_dir_path)
	mask_path_list = os.listdir(mask_dir_path)

	# Filter images with file extension being .png
	image_path_list = [img for img  in image_path_list if img.endswith('.png')]
	mask_path_list = [img for img  in mask_path_list if img.endswith('.png')]

	# Align inputs and masks
	image_path_list.sort()
	mask_path_list.sort()

	print(f"Number of images: {len(image_path_list)}")
	print(f"Number of masks: {len(mask_path_list)}")
	
	# FIVES Dataset
	for image_path, mask_path in zip(image_path_list, mask_path_list):
		if image_path.endswith('png'):
			print(image_path)
			assert os.path.basename(image_path)[:-4] == os.path.basename(mask_path)[:-4]
			_id = os.path.basename(image_path)[:-4]
			image_path = os.path.join(image_dir_path, image_path)
			mask_path = os.path.join(mask_dir_path, mask_path)
			image = cv2.imread(image_path, cv2.IMREAD_COLOR)
			mask = plt.imread(mask_path, cv2.IMREAD_COLOR)
			if mask.ndim == 3:
				mask = np.int64(np.all(mask[...,:3] == 1, axis=2))

			# Pre-processing with CLAHE Method
			new_image = clahe_equalized(image)
			new_mask = mask #clahe_equalized(mask)

			# Save to processed folder after pre-processing
			save_dir_path = processed_folder + '/Images'
			os.makedirs(save_dir_path, exist_ok=True)
			np.save(os.path.join(save_dir_path, _id + '.npy'), new_image)

			save_dir_path = processed_folder + '/Labels'
			os.makedirs(save_dir_path, exist_ok=True)
			np.save(os.path.join(save_dir_path, _id + '.npy'), new_mask)
