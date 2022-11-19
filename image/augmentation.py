import cv2
import numpy as np


class AugmentationPipeline:
  
  def __init__(self, x: int, y: int):
    self.x = x
    self.y = y
  
  def load_augmented_images(self, base_images: list) -> list:
      shifted_images = self._vertical_shift(base_images)
      bright_images = self._brightness(base_images)
      rotated_images = self._rotation(base_images)
      sobel_images = self._edge_detection(base_images)
      
      augmented_images = shifted_images + bright_images + rotated_images
      augmented_images = [cv2.resize(img, (self.x, self.y)) for img in augmented_images]
      
      return augmented_images
  
  def _vertical_shift(self, images):
    final_images = []
    for img in images:
      ratio = np.random.randint(0, 101) / 100
      ratio = np.random.uniform(-ratio, ratio)
      h, w = img.shape[:2]
      to_shift = h*ratio
      
      if ratio > 0:
          aug_img = img[:int(h-to_shift), :, :]
      
      if ratio < 0:
          aug_img = img[int(-1*to_shift):, :, :]

      final_images.append(aug_img)
    
    return final_images

  def _brightness(self, images: list) -> list:
    final_images = []
    for img in images:
      low = np.random.randint(0, 10)
      high = np.random.randint(11, 30)
      value = np.random.uniform(low, high)
      
      hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
      hsv = np.array(hsv, dtype = np.float64)
      hsv[:,:,1] = hsv[:,:,1]*value
      hsv[:,:,1][hsv[:,:,1]>255]  = 255
      hsv[:,:,2] = hsv[:,:,2]*value 
      hsv[:,:,2][hsv[:,:,2]>255]  = 255
      hsv = np.array(hsv, dtype = np.uint8)
      aug_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
      final_images.append(aug_img)
      
      
    return final_images

  def _rotation(self, images: list) -> list:
      final_images = []
      for img in images:
        low = np.random.randint(0, 10)
        high = np.random.randint(11, 30)
        angle = np.random.uniform(low, high)
        
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
        aug_img = cv2.warpAffine(img, M, (w, h))
        
        final_images.append(aug_img)
      
      return final_images
  
  def _edge_detection(self, images: list) -> list:
    final_images = []
    
    for img in images:
      img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
      aug_img = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
      final_images.append(aug_img)
    
    return final_images
      