import cv2 
import numpy as np 

class IntensityTransformer:

	def log_transform(self, image):
		
		c = 255/(np.log(1 + np.max(image))) 
		
		log_transformed = c * np.log(1 + image) 
		log_transformed = np.array(log_transformed, dtype = np.uint8) 
		
		return log_transformed

	def gamma_transform(self, image, gamma):

		gamma_corrected = np.array(255*(image / 255) ** gamma, dtype = 'uint8') 
		return gamma_corrected
