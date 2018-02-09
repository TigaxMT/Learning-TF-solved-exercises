import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
import tensorflow as tf
import os


print("EXERCISE 1", end='\n')
print("Combine the transposing code with the flip code to rotate clock wise.",end='\n')


# Fisrt , load the image
dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + "/MarshOrchid.jpg"

image = mpimg.imread(filename)

#Get height , width and depth of our image 
height , width , depth = image.shape

# Create a TensorFlow Variable
x = tf.Variable(image,name='x')

model = tf.global_variables_initializer()

# TRANSPOSE THE IMAGE


with tf.Session() as session:
	x = tf.transpose(x , perm=[1,0,2])

	session.run(model)

	result = session.run(x)

	height , width , depth = result.shape

	# I'm not 100% sure, but we use [width] because we want to start from the x axis from left to right. 
	# In this way we have access to the list of the value of each x of the image

	x = tf.reverse_sequence(x, [width] * height, 1 , batch_dim=0)

	# reverse_sequence(input, dimension of the image/array/etc, 1 for left to right or 0 for ignore axis x, batchdim=0 to the top to bottom or 1 for ignore axis y)

	result = session.run(x)

print(result.shape,end='\n')
plt.imshow(result)
plt.show()