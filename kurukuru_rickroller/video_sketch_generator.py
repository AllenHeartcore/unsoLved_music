import numpy as np
from scipy.signal import convolve2d
import cv2
from tqdm import tqdm

filename = 'nggyu_large.flv'
sigmoid_k = .025
sigmoid_b = 96
lower_cutoff = 24
masks = [np.array([(1, 0, -1), (1, 0, -1), (1, 0, -1)]), 
         np.array([(1, 1, 1), (0, 0, 0), (-1, -1, -1)]), 
         np.array([(2, 1, 0), (1, 0, -1), (0, -1, -2)]), 
         np.array([(0, 1, 2), (-1, 0, 1), (-2, -1, 0)])]

def sigmoid(x):
    return 1 / (1 + np.exp(sigmoid_k * (sigmoid_b - x)))

def sketch(image):
    gray = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
    bound = np.zeros_like(gray)
    for mask in masks:
        bound += abs(convolve2d(gray, mask, boundary='symm')[1:-1, 1:-1])
    bound = bound / bound.max() * 255
    bound = 255 * (sigmoid(bound) - sigmoid(0)) / (sigmoid(255) - sigmoid(0))
    bound = (bound <= lower_cutoff) * 255
    bound = np.stack([np.round(bound)] * image.shape[2], 2)
    return bound.astype(np.uint8)

capture = cv2.VideoCapture(filename)
filename = filename.split('.')
writer = cv2.VideoWriter('.'.join(filename[:-1]) + '_boundary.mp4', 
                         cv2.VideoWriter_fourcc(*'MP4V'), capture.get(cv2.CAP_PROP_FPS), 
                         (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))), True)
if capture.isOpened():
    for frame in tqdm(range(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))):
        _, img = capture.read()
        if frame % 2 == 0:
            img = sketch(img)
            writer.write(img)
writer.release()
