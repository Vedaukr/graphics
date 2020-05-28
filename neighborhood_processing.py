import cv2 
import numpy as np 
import scipy as sp
import scipy.ndimage

class NeighborhoodProcessing:

	def apply_median_filter(self, image):
		img_median = cv2.medianBlur(image, 25)
		return img_median

	def apply_gaussian_filter(self, image, sigma=2, truncate=4):
		gaussian_filtered = sp.ndimage.filters.gaussian_filter(image, sigma=sigma, truncate=truncate)
		return gaussian_filtered 