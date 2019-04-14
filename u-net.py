# coding: utf-8

# importing packages
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import numpy as np
from keras.models import Input, Model
from keras.layers import *
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array, load_img
import glob
import matplotlib.pyplot as plt


# Reading images and GT
'''
Please place your Data as well as Segmentation label in dataset folder
'''

path = '../dataset'
train_images = [] 
train_labels = []
test_images = []
test_labels = []
for f in os.listdir(path):
    if f.split('_')[0] not in ['10','11']:
        train_images.append(img_to_array(load_img(os.path.join(path,f,'CT.jpg'))))
        train_labels.append(img_to_array(load_img(os.path.join(path,f,'segmentation_mask.jpg'))))
    else:
        test_images.append(img_to_array(load_img(os.path.join(path,f,'CT.jpg'))))
        test_labels.append(img_to_array(load_img(os.path.join(path,f,'segmentation_mask.jpg'))))

train_images = np.array(train_images)[:,:,:,0:1].astype('float32')/255.0
train_labels = np.array(train_labels)[:,:,:,0:1]
test_images = np.array(test_images)[:,:,:,0:1].astype('float32')/255.0
test_labels = np.array(test_labels)[:,:,:,0:1]


# Reinitializing any label pixel value greater then 0 to 1 (Only for binary mask problem)
train_labels[train_labels>0]=1
test_labels[test_labels>0]=1


# Dimension of the train and test images and labels
print ('Train Images: {}'.format(train_images.shape))
print ('Train Labels: {}'.format(train_labels.shape))
print ('Test Images: {}'.format(test_images.shape))
print ('Test labels: {}'.format(test_labels.shape))


# Model Architecture (UNet)
def unet(size = (512,512,1)):
    input1 = Input(size)
    
    # Encoder
    # Block 1
    conv1= Conv2D(64, 3, activation = 'relu', padding= 'same', kernel_initializer='he_normal')(input1)
    conv1= Conv2D(64, 3, activation = 'relu', padding= 'same', kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1= MaxPooling2D(pool_size=(2,2))(conv1)
    
    # Block 2
    conv2= Conv2D(128, 3, activation = 'relu', padding= 'same', kernel_initializer='he_normal')(pool1)
    conv2= Conv2D(128, 3, activation = 'relu', padding= 'same', kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2= MaxPooling2D(pool_size=(2,2))(conv2)
    
    # Block 3
    conv3= Conv2D(256, 3, activation = 'relu', padding= 'same', kernel_initializer='he_normal')(pool2)
    conv3= Conv2D(256, 3, activation = 'relu', padding= 'same', kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3= MaxPooling2D(pool_size=(2,2))(conv3)
    
    # Block 4
    conv4= Conv2D(512, 3, activation = 'relu', padding= 'same', kernel_initializer='he_normal')(pool3)
    conv4= Conv2D(512, 3, activation = 'relu', padding= 'same', kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4= MaxPooling2D(pool_size=(2,2))(conv4)
    
    # Block 5
    conv5= Conv2D(1024, 3, activation = 'relu', padding= 'same', kernel_initializer='he_normal')(pool4)
    conv5= Conv2D(1024, 3, activation = 'relu', padding= 'same', kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    
    # Decoder
    # Block 6
    up6 = Conv2D(512, 2, activation = 'relu', padding= 'same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv5))
    merge6 = concatenate([conv4, up6], axis=3)
    conv6= Conv2D(512, 3, activation = 'relu', padding= 'same', kernel_initializer='he_normal')(merge6)
    conv6= Conv2D(512, 3, activation = 'relu', padding= 'same', kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)
    
    # Block 7
    up7 = Conv2D(256, 2, activation = 'relu', padding= 'same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7= Conv2D(256, 3, activation = 'relu', padding= 'same', kernel_initializer='he_normal')(merge7)
    conv7= Conv2D(256, 3, activation = 'relu', padding= 'same', kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
    
    # Block 8
    up8 = Conv2D(128, 2, activation = 'relu', padding= 'same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8= Conv2D(128, 3, activation = 'relu', padding= 'same', kernel_initializer='he_normal')(merge8)
    conv8= Conv2D(128, 3, activation = 'relu', padding= 'same', kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)
    
    # Block 9
    up9 = Conv2D(64, 2, activation = 'relu', padding= 'same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9= Conv2D(64, 3, activation = 'relu', padding= 'same', kernel_initializer='he_normal')(merge9)
    conv9= Conv2D(64, 3, activation = 'relu', padding= 'same', kernel_initializer='he_normal')(conv9)
    
    conv9 = Conv2D(2, 3, activation = 'relu', padding= 'same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    
    model = Model(inputs=input1, outputs=conv10)
    
    # Compiling the model
    model.compile(optimizer=Adam(lr=1e-04), loss= 'binary_crossentropy', metrics = ['accuracy'])
    
    # Displaying the model architecture
    #model.summary()
    
    return model


def training(model, X_train, Y_train, BATCH_SIZE, epoch):
    # We can create checkpoint as well as early stopping here which I have not done
    
    # training network
    model.fit(x= X_train, 
             y= Y_train,
             batch_size = BATCH_SIZE,
             epochs = epoch,
             shuffle=True,
             verbose=2)
    
    return model


def IoU(Pred, GT, num_of_classes):
    class_intersection = np.zeros(num_of_classes)
    class_union = np.zeros(num_of_classes)
    for i in range(num_of_classes):
        class_intersection[i] = np.float32(np.sum(Pred==GT)*(GT==i))
        class_union[i] = np.sum(GT=i)+np.sum(Pred=i)-class_intersection[i]
    return class_intersection, class_union


# Driver code for training
batch_size = 2
epoch = 1
model = unet()
model = training(model, train_images, train_labels, batch_size, epoch)


# testing model
# (Basic)
model.evaluate(test_images, test_labels, verbose=1)

# (USING IoU)
predicted_lables = model.predict(test_images, verbose=1)
predicted_lables = (predicted_lables >0.5).astype(np.uint8)

class_intersection, class_union = IoU(predicted_lables, test_labels, 2)

bg = class_intersection[0]/class_union[0]
obj = class_intersection[1]/class_union[1]
print ('IoU Score of Background: {}'.format(bg))
print ('IoU Score of object: {}'.format(obj))
print ('Mean IoU Score: {}'.format(np.mean([bg,obj]))