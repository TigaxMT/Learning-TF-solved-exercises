import tensorflow as tf 
import numpy as np 
import resource


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


print("EXERCISE 1",end='\n\n')
print("Create a large matrix (at least 10,000,000) of integer values (for example, use NumPy’s randint function). Check the memory usage after the matrix is created. Then, convert the matrix to float values using TensorFlow’s to_float function. Check the memory usage again to see an increase in memory usage of more than double. The “doubling” is caused by a copy of the matrix being created, but what is the cause of the “extra increase”?",end='\n\n')

session = tf.InteractiveSession()

tensor_arr = tf.constant(np.random.randint(10000000))

#print(tensor_arr.eval())

print(convert_KB(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

tensor_arr = tf.to_float(tensor_arr)

#print(tensor_arr.eval())

print(convert_KB(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

session.close()