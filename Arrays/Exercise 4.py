import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
import tensorflow as tf
import os
#import numpy as np


print("EXERCISE 4", end='\n')
print("Compute a “mirror”, where the first half of the image is copied, flipped (l-r) and then copied into the second half.",end='\n')


# Fisrt , load the image
dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + "/MarshOrchid.jpg"

image = mpimg.imread(filename)

#Get height , width and depth of our image 
height , width , depth = image.shape

x = tf.Variable(image,name='x')

model = tf.global_variables_initializer()

with tf.Session() as session:
	session.run(model)
	
	# I'm not 100% sure, but we use [width] because we want to start from the x axis from left to right. 
	# In this way we have access to the list of the value of each x of the image
	
	left_part = tf.slice(x, [0, 0, 0], [int(height), int(width/2), int(depth)]) #Extract Left part
	rigth_part = tf.slice(x, [0, int(width/2), 0], [int(height), int(width/2), int(depth)]) #Extract Right part
	
	left_part = tf.reverse_sequence(left_part, int(height) * [int(width/2)], 1, batch_dim=0) #Reverse Left Part
	rigth_part = tf.reverse_sequence(left_part, int(height) * [int(width/2)], 1, batch_dim=0) #Reverse Right Part
	

	# Other way to write the 2 above lines of code. Note: import the numpy
	#left_part = tf.reverse_sequence(left_part, np.ones((height,)) * width/2, 1, batch_dim=0) #Reverse Left Part
    #rigth_part = tf.reverse_sequence(left_part, np.ones((height,)) * width/2, 1, batch_dim=0) #Reverse Right Part


	x = tf.concat([left_part, rigth_part],1) #Concat them together along the second edge (the width)
	
	result = session.run(x)

plt.imshow(result)
plt.show()