import os
import csv
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sklearn

samples = []

with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)

# ignore the header of the table
samples = samples[1:]
# print(len(samples))

train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			measurements = []
			for batch_sample in batch_samples:
				for i in range(3):
					source_path = batch_sample[i]
					filename = source_path.split('/')[-1]
					current_path = './data/IMG/' + filename

					image = mpimg.imread(current_path) # image = cv2.imread(current_path)
					images.append(image)
					# augment data by flipping images
					image_flipped = np.fliplr(image)
					images.append(image_flipped)
					
					measurement = float(batch_sample[3])
					correction = 0.2 # this is a parameter to tune
					if i == 0:
						measurements.append(measurement)
					elif i == 1:
						measurements.append(measurement + correction)
					elif i == 2:
						measurements.append(measurement - correction)
					# augment data by flipping images
					measurement_flipped = -measurement
					if i == 0:
						measurements.append(measurement_flipped)
					elif i == 1:
						measurements.append(measurement_flipped - correction)
					elif i == 2:
						measurements.append(measurement_flipped + correction)
				

            # trim image to only see section with road
			X_train = np.array(images)
			y_train = np.array(measurements)
			yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()

model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D( cropping = ((70,25),(0,0)) ))

model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation = 'relu')) # conv_1
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation = 'relu')) # conv_2
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation = 'relu')) # conv_3
model.add(Convolution2D(64, 3, 3, activation = 'relu')) # conv_4

model.add(Flatten())
model.add(Dense(100)) # fc_1
model.add(Dropout(0.4))
model.add(Dense(50)) # fc_2
model.add(Dropout(0.4))
model.add(Dense(16)) # fc_3
model.add(Dense(1))


model.compile(loss = 'mse', optimizer = 'adam')

model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=10)


model.save('model.h5')


