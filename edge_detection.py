import cv2 
import numpy as np 

class EdgeDetector:

	def detect_edges(self, image):
	    
	    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	    blur = cv2.GaussianBlur(gray, (5, 5), 0)
	    canny = cv2.Canny(blur, 10, 70)
	    _, mask = cv2.threshold(canny, 70, 255, cv2.THRESH_BINARY)
	    
	    return mask
