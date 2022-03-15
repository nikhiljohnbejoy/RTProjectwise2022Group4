#For image data preprocessing
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dropout,Conv2D,Flatten,Dense, MaxPooling2D

#ImageDataGenerator generates batches of tensor image data with real-time data augmentation
generator=image.ImageDataGenerator(rescale=1./255)
BatchSize= 32
TargetSize=(24,24)
#Takes the path to training directory & generates batches of augmented data.
train_batch= generator.flow_from_directory('eyes_dataset/train',batch_size=BatchSize,shuffle=True,color_mode='grayscale',class_mode='categorical', target_size=TargetSize)
#Takes the path to validation directory & generates batches of augmented data.
valid_batch= generator.flow_from_directory('eyes_dataset/test',batch_size=BatchSize,shuffle=True,color_mode='grayscale',class_mode='categorical', target_size=TargetSize)
SPE= len(train_batch.classes)//BatchSize
VS = len(valid_batch.classes)//BatchSize

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24,24,1)),
#downscale the image and replace it with Convolution    
    MaxPooling2D(pool_size=(1,1)),
    Conv2D(32,(3,3),activation='relu'),
    MaxPooling2D(pool_size=(1,1)),
#Convolution filters used each of size 3x3
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1,1)),
#Prevent overfitting
    Dropout(0.25),
#flattening for a classification output
    Flatten(),
#Densely-connected NN layer, 128 units as output
    Dense(128, activation='relu'),
#Applies Dropout to the input.
    Dropout(0.5),
#Softmax activation funciton in the output layer
    Dense(2, activation='softmax')
])
#Configuring the model for training
#Optimizer used: Adam algorithm, loss used: crossentropy loss between the labels and predictions
#Metrics used: evaluation of accuracy
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#Fits the model on data yielded batch-by-batch
model.fit_generator(train_batch, validation_data=valid_batch,epochs=15,steps_per_epoch=SPE ,validation_steps=VS)
#Saving to a HDF5 file
print('NUMBER OF LAYERS',len(model.layers))
model.save('model/eye_model.h5', overwrite=True)
