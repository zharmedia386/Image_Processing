import numpy as npy
 
# using matplotlib
import matplotlib.image as img
 
# using statistics to import mean
# for mean calculation
from statistics import mean
 
m = img.imread("azhar_pic.jpg")
 
# determining width and height of original image
w, h = m.shape[:2]
 
# new Image dimension with 4 attribute in each pixel
newImage = npy.zeros([w, h, 4])
print( w )
print( h )
 
for i in range(w):
   for j in range(h):
      # ratio of RGB will be between 0 and 1
      lst = [float(m[i][j][0]), float(m[i][j][1]), float(m[i][j][2])]
      avg = float(mean(lst))
      newImage[i][j][0] = avg
      newImage[i][j][1] = avg
      newImage[i][j][2] = avg
      newImage[i][j][3] = 1 # alpha value to be 1
 
# Save image using imsave
img.imsave('grayedImage.png', newImage)