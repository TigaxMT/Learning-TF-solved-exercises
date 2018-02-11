import tensorflow as tf 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import os

print("EXERCISE 2", end='\n')
print("Break the image apart into four “corners”, then stitch it back together again.")


# First , load the image
dir_path = os.path.dirname(os.path.realpath(__file__))

filename = dir_path + "/MarshOrchid.jpg"
raw_image_data = mpimg.imread(filename)

#Get height , width and depth of our image 
height , width , depth= raw_image_data.shape

half_width = int(width/2)
half_height = int(height/2)

image = tf.placeholder("uint8", [None,None,None])

left_top = tf.slice(image,[0,0,0], [half_height,half_width,-1])
right_top = tf.slice(image,[0,half_width,0], [half_height,-1,-1])

left_bottom = tf.slice(image,[half_height,0,0], [-1,half_width,-1])
right_bottom = tf.slice(image,[half_height,half_width,0], [-1,-1,-1])

with tf.Session() as session:
	result_1 = session.run(left_top, feed_dict={image: raw_image_data})
	result_2 = session.run(right_top, feed_dict={image: raw_image_data})
	result_3 = session.run(left_bottom, feed_dict={image: raw_image_data})
	result_4 = session.run(right_bottom, feed_dict={image: raw_image_data})


	# Here I can't concatenate the corners because they don't have the same size on width, probably because I needed
	# to cast them to int. If you know how solve this problem please tell us or pull request your solution.

	#top = tf.concat([result_1,result_2],0)
	#bottom = tf.concat([result_3,result_4],0)
	#result = tf.concat([top,bottom],1)

plt.imshow(result_1)
plt.title("Left_Top")
plt.show()

plt.imshow(result_2)
plt.title("Right_Top")
plt.show()

plt.imshow(result_3)
plt.title("Left_Bottom")
plt.show()

plt.imshow(result_4)
plt.title("Right_Bottom")
plt.show()

#plt.imshow(result)
#plt.title("stitch it back together again")
#plt.show()
