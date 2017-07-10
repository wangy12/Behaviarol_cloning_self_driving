import csv
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

lines = []

with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)
		
		
images = []
measurements = []
for line in lines[1:]:
	for i in range(3):
		source_path = line[i]
		filename = source_path.split('/')[-1]
		current_path = './data/IMG/' + filename
		image = mpimg.imread(current_path) # image = cv2.imread(current_path)
		images.append(image)
		image_flipped = np.fliplr(image)
		images.append(image_flipped)
	
		measurement = float(line[3])
		correction = 0.2 # this is a parameter to tune
		if i == 0:
			measurements.append(measurement)
		elif i == 1:
			measurements.append(measurement + correction)
		elif i == 2:
			measurements.append(measurement - correction)
		
		measurement_flipped = -measurement
		if i == 0:
			measurements.append(measurement_flipped)
		elif i == 1:
			measurements.append(measurement_flipped - correction)
		elif i == 2:
			measurements.append(measurement_flipped + correction)
			
# print( len(measurements) )

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()

model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D( cropping = ((70,25),(0,0)) ))

model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation = 'relu')) # conv_1 
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation = 'relu')) # conv_2 
model.add(Dropout(0.3))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation = 'relu')) # conv_3 
model.add(Dropout(0.3))
model.add(Convolution2D(64, 3, 3, activation = 'relu')) # conv_4 
model.add(Dropout(0.3))
model.add(Convolution2D(64, 3, 3, activation = 'relu')) # conv_5


model.add(Flatten())
model.add(Dense(100)) # fc_1
model.add(Dropout(0.4))
model.add(Dense(50)) # fc_2
model.add(Dropout(0.3))
model.add(Dense(16)) # fc_3
model.add(Dense(1))


model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 3)

model.save('model.h5')

'''
history_object = model.fit_generator(train_generator, samples_per_epoch =
    len(train_samples), validation_data = 
    validation_generator,
    nb_val_samples = len(validation_samples), 
    nb_epoch=3, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
'''
