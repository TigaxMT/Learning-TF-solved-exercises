import tensorflow as tf 
import resource
from PIL import Image
from io import BytesIO
import matplotlib.image as mpimg
import os


# This function will convert the info of the used RAM for GB or MB 
def convert_KB(val):

	result = 0.0

	# Convert KB to GB
	if val >= 1048576:
		result = val/1048576
		return "Used {}GB of RAM".format(round(result,2))
	
	#Converto KB to MB
	else:
		result = val/1024
		return "Used {}MB of RAM".format(round(result,2))


print("EXERCISE 2",end='\n\n')
print("Use TensorFlow’s image functions to convert the image from the previous tutorial (or another image) to JPEG with different functions and record the memory usage.",end='\n\n')

session = tf.InteractiveSession()

# Fisrt , load the image
dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + "/MarshOrchid.jpg"
raw_image_data = mpimg.imread(filename)

# Create a Tensor with the image data
img = tf.constant(raw_image_data)

print(convert_KB(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

# Now evaluate the data, that is , tell to tensoflow to do the needed opertions to give us a
# result(the real image data)
img = img.eval()

print(convert_KB(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


# Use the built function of tf to enconde to jpeg (The image already is jpeg
# but we enconde the same)
jpeg = tf.image.encode_jpeg(img)


# .eval() in InteractiveSession is the same of session.run() on tf.Session()
jpeg = jpeg.eval()

print(convert_KB(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

# read data from bytes
im = Image.open(BytesIO(jpeg))

# save the enconded image with pillow module
im.save("MarshOrchid_enconded.jpg")

print(convert_KB(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

session.close()