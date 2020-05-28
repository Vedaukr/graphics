import cv2 
import numpy as np 
import scipy as sp
import scipy.ndimage

from intensity_transformer import IntensityTransformer
from edge_detection import EdgeDetector
from neighborhood_processing import NeighborhoodProcessing

edge_detector = EdgeDetector()
intensity_transformer = IntensityTransformer()
neighborhood_processor = NeighborhoodProcessing()

# intensity transform
img = cv2.imread('./intencity/original.jpg')
log_transformed = intensity_transformer.log_transform(img)

gamma_corrected_08 = intensity_transformer.gamma_transform(img, gamma=0.8)
gamma_corrected_12 = intensity_transformer.gamma_transform(img, gamma=1.2)
gamma_corrected_18 = intensity_transformer.gamma_transform(img, gamma=1.8)

cv2.imwrite('./intencity/log_transformed.jpg', log_transformed)
cv2.imwrite('./intencity/gamma_corrected (0.8).jpg', gamma_corrected_08) 
cv2.imwrite('./intencity/gamma_corrected (1.2).jpg', gamma_corrected_12) 
cv2.imwrite('./intencity/gamma_corrected (1.8).jpg', gamma_corrected_18) 

# edges detection
img = cv2.imread('./edge/original.jpg')
edges = edge_detector.detect_edges(img)
cv2.imwrite('./edge/edges.jpg', edges)

# neighborhood processing
img = cv2.imread('./neighborhood/original.jpg')
median_filtered = neighborhood_processor.apply_median_filter(img)
gaussian_filtered = neighborhood_processor.apply_gaussian_filter(img)
cv2.imwrite('./neighborhood/median_filtered.jpg', median_filtered)
cv2.imwrite('./neighborhood/gaussian_filtered.jpg', gaussian_filtered)