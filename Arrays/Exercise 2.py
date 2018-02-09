import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
import tensorflow as tf
import os


print("EXERCISE 2", end='\n')
print("Currently, the flip code (using reverse_sequence) requires width to be precomputed. Look at the documentation for the tf.shape function, and use it to compute the width of the x variable within the session.",end='\n')


# Fisrt , load the image
dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + "/MarshOrchid.jpg"

image = mpimg.imread(filename)

#Get height , width and depth of our image 
height , width , depth = image.shape


# Create a TensorFlow Variable

x = tf.Variable(image, name='x')

model = tf.global_variables_initializer()

with tf.Session() as session:
	session.run(model)

	#tf.shape(x) return height[index 0] , width[index 1] and depth[index 2] 

	width = session.run(tf.shape(x)[1])
	height = session.run(tf.shape(x)[0])

	# I'm not 100% sure, but we use [width] because we want to start from the x axis from left to right. 
	# In this way we have access to the list of the value of each x of the image

	x = tf.reverse_sequence(x, [width] * height, 1 , batch_dim=0)
	
	result = session.run(x)

plt.imshow(result)
plt.show()