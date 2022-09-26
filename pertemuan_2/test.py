
# Import the necessary libraries
from PIL import Image
from numpy import asarray
  
  
# load the image and convert into 
# numpy array
img = Image.open('lena2.png')
numpydata = asarray(img)
  
# data
print(numpydata)