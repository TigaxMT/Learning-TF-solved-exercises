import tensorflow as tf 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import os

# This exercise aren't 100% solved yet , because I can't show the image converted to grayscale if you know how solved it , please
# pull request your solution

print("EXERCISE 3", end='\n')
print(" Convert the image into grayscale. One way to do this would be to take just a single colour channel and show that. Another way would be to take the average of the three channels as the gray colour.")

# First , load the image
dir_path = os.path.dirname(os.path.realpath(__file__))

filename = dir_path + "/MarshOrchid.jpg"
raw_image_data = mpimg.imread(filename)

image = tf.placeholder("float32", [None,None,None])
to_grayscale = tf.slice(image, [0,0,0], [-1,-1,-1])

with tf.Session() as session:
	result = session.run(to_grayscale, feed_dict={image: raw_image_data})

	grayscale = tf.image.rgb_to_grayscale(result)
	grayscale = tf.image.convert_image_dtype(grayscale,tf.float32)

	print(grayscale.shape)

plt.imshow(grayscale, cmap="gray")
plt.show()