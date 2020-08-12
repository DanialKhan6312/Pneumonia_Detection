from tensorflow.keras.preprocessing.image import, ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv1D,MaxPooling2D,Dropout,BatchNormalization,AveragePooling2D
import matplotlib.pyplot as plt


###DATA PREP###

train_path = './chest_xray/train'
val_path = './chest_xray/val'
test_path = './chest_xray/test'
train_batch = ImageDataGenerator(rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        validation_split=0.2,
        ).flow_from_directory(train_path,target_size = (100,100),classes =['NORMAL','PNEUMONIA'] ,batch_size=34,color_mode="rgb",subset="training")
test_batch = ImageDataGenerator(rescale=1/255).flow_from_directory(test_path,target_size = (100,100),classes =['NORMAL','PNEUMONIA'] ,color_mode="rgb")
val_batch = ImageDataGenerator(rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
        ,validation_split=0.2).flow_from_directory(train_path,target_size = (100,100),classes =['NORMAL','PNEUMONIA'],batch_size=4,color_mode="rgb",subset="validation")

###MODEL ARCHITECTURE###
my_net=Sequential()
my_net.add(Conv2D(16, (3, 3), activation="relu", input_shape=(100,100,3)))
my_net.add(Dropout(0.3))
my_net.add(AveragePooling2D())
my_net.add(Conv2D(32, (3, 3), activation="relu" ))
my_net.add(Dropout(0.3))
my_net.add(AveragePooling2D())
my_net.add(Flatten())
my_net.add(Dense(512, activation = 'relu'))
my_net.add(Dropout(0.6))
my_net.add(Dense(2,activation = 'softmax'))
my_net.summary()
my_net.compile(loss = 'binary_crossentropy', optimizer = 'adam',metrics=['accuracy'])

###TRAINING###

checkpoint = ModelCheckpoint("./Model", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
my_net.fit(train_batch,epochs = 1,callbacks=callbacks_list,validation_data = val_batch)

###TRAINING ANALYTICS###
history = my_net.fit(train_batch,epochs = 5,validation_data=val_batch,validation_steps=11)

#printing overall accuracy

test_accu = my_net.evaluate_generator(test_batch)
print("Test accuracy is: ", test_accu[1]*100, '%')

#graphs

#accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.show()

#loss
plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Test set'], loc='upper left')
plt.show()