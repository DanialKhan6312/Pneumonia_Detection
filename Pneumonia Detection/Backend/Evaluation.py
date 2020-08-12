from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report

### DATA PREP ###

test_path = './chest_xray/test'
test_batch = ImageDataGenerator(rescale=1/255).flow_from_directory(test_path,target_size = (100,100),classes =['NORMAL','PNEUMONIA'] ,color_mode="rgb")

### LOAD MODEL ###

my_net = load_model("Model")

### TESTING AND METRICS ###
y_img_batch, y_class_batch = test_batch[0]
y_pred = np.argmax(my_net.predict_generator(y_img_batch),-1)
y_true = np.argmax(y_class_batch,-1)
print (my_net.evaluate(test_batch))
print(classification_report(y_true, y_pred))
