""" Replicating the SimpleNet model architecture. """

import keras
import numpy as np
import keras.backend as K
from keras.utils import np_utils, plot_model
from keras.datasets import cifar10
from keras.models import Sequential
from keras.models import load_model, Model
from keras import regularizers, optimizers
from keras.layers import Conv2D, MaxPooling2D, Input, concatenate, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.initializers import glorot_normal, RandomNormal, Zeros
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization

# Data Retrieval & mean/std preprocess
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

mean = np.mean(x_train,axis=(0,1,2,3))
std = np.std(x_train,axis=(0,1,2,3))
x_train = (x_train-mean)/(std+1e-7)
x_test = (x_test-mean)/(std+1e-7)

num_classes = 10
y_train = np_utils.to_categorical(y_train,num_classes)
y_test = np_utils.to_categorical(y_test,num_classes)


# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
    )
datagen.fit(x_train)

# Define Model architecture
def create_model(s = 2, weight_decay = 1e-2):
    input_layer = Input((32, 32, 3))
    model = input_layer
    act = 'relu'
	
    tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(model)
    tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)
    tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(model)
    tower_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(tower_2)
    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(model)
    tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)
    tower_4 = Conv2D(64, (1, 1), padding='same', activation='relu')(model)
    model = concatenate([tower_1, tower_2, tower_3, tower_4], axis=3)

    model = Conv2D(32, (4,4), strides=1, padding='same', kernel_initializer=glorot_normal())(model)
    model = AveragePooling2D(pool_size=(3,3), strides=2)(model)
    model = BatchNormalization()(model)
    model = Flatten()(model)
    model = Dense(num_classes, activation='softmax')(model)
    return Model(inputs=input_layer, outputs=model)

if __name__ == "__main__":
	# Prepare for training
	model = create_model()
	batch_size = 120
	epochs = 25
	train = {}

	plot_model(model, to_file='full_cifar100_result.png')
	
	# First training for 50 epochs - (0-50)
	opt_adm = keras.optimizers.Adadelta()
	model.compile(loss='categorical_crossentropy', optimizer=opt_adm, metrics=['accuracy'])
	train["part_1"] = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
										steps_per_epoch=x_train.shape[0] // batch_size,epochs=epochs,
										verbose=1,validation_data=(x_test,y_test))
	model.save("simplenet_generic_first.h5")
	print(train["part_1"].history)

	# Training for 25 epochs more - (50-75)
	opt_adm = keras.optimizers.Adadelta(lr=0.7, rho=0.9)
	model.compile(loss='categorical_crossentropy', optimizer=opt_adm, metrics=['accuracy'])
	train["part_2"] = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
										steps_per_epoch=x_train.shape[0] // batch_size,epochs=epochs,
										verbose=1,validation_data=(x_test,y_test))
	model.save("simplenet_generic_second.h5")
	print(train["part_2"].history)

	# Training for 25 epochs more - (75-100)
	opt_adm = keras.optimizers.Adadelta(lr=0.5, rho=0.85)
	model.compile(loss='categorical_crossentropy', optimizer=opt_adm, metrics=['accuracy'])
	train["part_3"] = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
										steps_per_epoch=x_train.shape[0] // batch_size,epochs=epochs,
										verbose=1,validation_data=(x_test,y_test))
	model.save("simplenet_generic_third.h5")
	print(train["part_3"].history)

	# Training for 25 epochs more  - (100-125)
	opt_adm = keras.optimizers.Adadelta(lr=0.3, rho=0.75)
	model.compile(loss='categorical_crossentropy', optimizer=opt_adm, metrics=['accuracy'])
	train["part_4"] = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
										steps_per_epoch=x_train.shape[0] // batch_size,epochs=epochs,
										verbose=1,validation_data=(x_test,y_test))
	model.save("simplenet_generic_fourth.h5")
	print(train["part_4"].history)

	print("\n \n Final Logs: ", train)
