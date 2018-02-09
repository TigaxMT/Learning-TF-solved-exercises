import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
import tensorflow as tf
import os


print("EXERCISE 3", end='\n')
print("Perform a “flipud”, which flips the image top-to-bottom.",end='\n')


# Fisrt , load the image
dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + "/MarshOrchid.jpg"

image = mpimg.imread(filename)

#Get height , width and depth of our image 
height , width , depth = image.shape

x = tf.Variable(image, name='x')

model = tf.global_variables_initializer()

with tf.Session() as session:
	session.run(model)

	width = session.run(tf.shape(x)[1])
	height = session.run(tf.shape(x)[0])

	# I'm not 100% sure, but we use [height] because we want to start from the y axis from top to bottom. 
	# In this way we have access to the list of the value of each y of the image
	x = tf.reverse_sequence(x, width * [height], 0 , batch_dim=1)
	
	result = session.run(x)

plt.imshow(result)
plt.show()