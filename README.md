
# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset

Digit classification and to verify the response for scanned handwritten images.

The MNIST dataset is a collection of handwritten digits. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively. The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. Here we build a convolutional neural network model that is able to classify to it's appropriate numerical value.

## Neural Network Model



## DESIGN STEPS

### STEP 1:
Import tensorflow and preprocessing libraries.

### STEP 2:
Download and load the dataset

### STEP 3:
Scale the dataset between it's min and max values

### STEP 4:
Using one hot encode, encode the categorical values

### STEP 5:
Split the data into train and test

### STEP 6:
Build the convolutional neural network model

### STEP 7:
Train the model with the training data

### STEP 8:
Plot the performance plot

### STEP 9:
Evaluate the model with the testing data

### STEP 10:
Fit the model and predict the single input

## PROGRAM
### Name:Vijayaraj V
### Register Number:212222230174
```python

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(x_train,y_train),(x_test,y_test)=mnist.load_data()

x_train.shape

x_test.shape


singleimage=x_train[200]


singleimage.shape

plt.imshow(singleimage)

y_train.shape

x_train.min()

x_train.max()

x_train_scaled=x_train/255
x_test_scaled=x_test/255

x_train_scaled.min()


x_test_scaled.max()

y_train[0]

y_train_ohe=utils.to_categorical(y_train,10)
y_test_ohe=utils.to_categorical(y_test,10)

y_train_ohe.shape

single_image = x_train[500]
plt.imshow(single_image,cmap='gray')

y_train_ohe[500]

X_train_scaled = x_train_scaled.reshape(-1,28,28,1)
X_test_scaled = x_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=16, kernel_size=(9,9), activation='relu'))
model.add(layers.MaxPool2D(pool_size=(3,3)))
model.add(layers.Flatten())
model.add(layers.Dense(65,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')

model.fit(X_train_scaled ,y_train_ohe, epochs=5,batch_size=64,validation_data=(X_test_scaled,y_test_ohe))

metrics = pd.DataFrame(model.history.history)

metrics.head()

print("Name:vijayaraj v Reg.No:212222230174 ")
metrics[['accuracy','val_accuracy']].plot()

print("Name:vijayaraj v Reg.No:212222230174 ")
metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

print("Name:vijayaraj v Reg.No:212222230174 ")
print(confusion_matrix(y_test,x_test_predictions))

print("Name:vijayaraj v Reg.No:212222230174 ")
print(classification_report(y_test,x_test_predictions))

img = image.load_img('9.png')

img

img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

x_single_prediction = np.argmax(model.predict(img_28_gray_scaled.reshape(1,28,28,1)),axis=1)

print(x_single_prediction)

plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0

x_single_prediction = np.argmax(model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),axis=1)

print(x_single_prediction)

```


## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot:

![image](https://github.com/vijayarajv1704/mnist-classification/assets/121303741/10946613-bc4c-43ef-b28a-64f6602a4420)


![image](https://github.com/vijayarajv1704/mnist-classification/assets/121303741/12fab053-97c3-48ca-80b7-f47dbd7b5386)


### Classification Report:

![image](https://github.com/vijayarajv1704/mnist-classification/assets/121303741/a465b6fd-7ba6-45f4-b08f-de2f1d44d2a4)



### Confusion Matrix:

![image](https://github.com/vijayarajv1704/mnist-classification/assets/121303741/5f59f081-b061-4270-ab2d-7542bda8c269)


### New Sample Data Prediction:

![image](https://github.com/vijayarajv1704/mnist-classification/assets/121303741/fec34469-116a-4dda-9609-ccb5473f7006)


![image](https://github.com/vijayarajv1704/mnist-classification/assets/121303741/9d684335-536c-496c-89a2-403739e12715)




## RESULT
A convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is developed sucessfully.
