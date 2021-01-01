import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

#device_name= tf.test.gpu_device_name()
#if device_name !='device:GPU:0':
#    raise SystemError("GPU Device Not Found")
#print('Found GPU at: {}'.format(device_name))


import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

#Part 1 - Data Preprocessing
#Generating images for the Training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

#Generating images for the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)

#Creating the Training set
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

#creating_Testing set
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
#Part 2 - Building the CNN
cnn = tf.keras.models.Sequential()

#step_1 : Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[64, 64, 3]))
#pading define the boundary and kernel define the row and column

#step_2: Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

#add_more layer for increasing the accuracy of test set and less gap between test and train
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

#when add third_time then filter will be 64

#step_2 Flatteing
cnn.add(tf.keras.layers.Flatten())

#step_4 full connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

#step_5 Output_layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#compiler CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Training the CNN on the Training set and evaluating it on the Test set
cnn.fit_generator(training_set,
                  steps_per_epoch = 8000,
                  epochs = 5,
                  validation_data = test_set,
                  validation_steps = 2000)


#val accuracy define the test set accuracy


