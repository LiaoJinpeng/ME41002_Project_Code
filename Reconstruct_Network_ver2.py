# -*- coding: utf-8 -*-

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import pathlib

class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
 
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
history = LossHistory()
class Variables():
    def __init__(self):
        """
        Definiton of the contast in the Network
        Args:
            lr_size: The Image Size of the Mosaic Image
            hr_size: The Image Size of the Reconstructed Image
            channel: The Image Channel of the Image (channel=3 for RGB image)
            epoch: The epoch of the training (Normally, more than 1000 training epoch is better)
            batch_size: The batch size in each epochs (if the GPU has high performance, the batch size should larger)
            number_of_datasets: The amount of the training datasets
            datasets_filepath: The filepath of the training datasets (Mosaic Image Datasets)
            testing_datasets: The filepath of the testing datasets, use to validate the result (High-resolution Datasets)
            using_cache_tensor: Using the cache can be faster, but need more Memory in the GPU
        """
        self.lr_size = 256 # Mosiac Image Size, Square image
        self.hr_size = 256 # High resolution image size 
        self.channel = 3 
        self.epoch = 2500
        self.batch_size = 3
        self.number_of_datasets = 70 # For example, you have 50 images for train, 
        self.datasets_filepath = pathlib.Path(r'C:\Users\Jinpeng Liao\Desktop\Train')
        self.testing_datasets = pathlib.Path(r'C:\Users\Jinpeng Liao\Desktop\Validation')
class Jpg_Tensor(): 
    def __init__(self, file_path, Image_Number=10,ishape=64,Channel=3):
        """ 
        This Class has Build in Function(BIF) to process the Original Image to Tensor Type
        The Tensor type datasets can be the input datasets of the Network training
        Args:
            file_path: strings type, the file path of the image datasets
            using_cache: Bool type, using the cache can be faster, but need more Memory
            Image_Number: Int type, It means the number of images to create the Tensor
            ishape: Int type, the output image shape will be ishape*ishape
            Channel: The Image Channel of the Image (channel=3 for RGB image)
        Return:
            The tensor type datasets
        """
        self.ishape = ishape
        self.channel = Channel
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.Image_Number = Image_Number
        self.data_path = file_path
        # Obtain the Image Path
        self.image_path = list(self.data_path.glob('*.jpg')) 
        self.all_image_paths = [str(path) for path in self.image_path]
        self.image_count = len(self.all_image_paths)
        # Preprocess the Image to the Tensor Type
        self.path_ds = tf.data.Dataset.from_tensor_slices(self.all_image_paths)
        self.image_ds = self.path_ds.map(self.load_and_preprocess_image)
        # Using Cache can improve the speed, but it will waste your RAM
        self.using_cache = False
        
    def preprocess_image(self,image):
        """
        This BIF can resize the image and decode the image to jpeg type
        """
        # Use the build in function of the Tensorflow to decode the JPEG
        image = tf.image.decode_jpeg(image, channels=self.channel)
        # Use the tensorflow.image.resize function to resize the iamge.
        image = tf.image.resize(image, [self.ishape, self.ishape],method='bicubic')
        # Normalization processing, make sure the value of the image is between 0 and 1
        image /= 255.0
        return image
    def load_and_preprocess_image(self,path):
        """
        This BIF can read the image from file_path and return preprocessed image
        """
        # Read the Image from the specific filepath
        image = tf.io.read_file(path)
        return self.preprocess_image(image)
    
    def return_tensor(self):
        """
        This BIF can return the Tensor type Datasets for Model Training
        """
        if self.using_cache:
            ds = self.image_ds.cache().repeat()
            ds = ds.batch(self.Image_Number)
        else:
            ds = self.image_ds
            ds = ds.repeat()
            ds = ds.batch(self.Image_Number)
        image_batch = next(iter(ds))
        return image_batch
class Plot():
    def __init__(self,test_datasets,original_datasets,model,number_of_image):
        """
        This function aims to plot the image result
        Args:
            test: Tensor type, the testing datasets of the Network
            train: Tensor type, the ground-true datasets of the Network
            model: tf.keras.Sequential type, The Single Image Super Resolution Model
            number: The number of the image, which will be plotted.
        """
        self.test = test_datasets
        self.train = original_datasets
        self.model = model
        self.number = number_of_image
    def Image_SR(self,num):
        """
        This function aims to return a reconstructed image
        Args:
            num: int type, the number of the processed image
        Return:
            The reconstructed super-resolution image
        """
        img_lr = self.test[num]
        # Reshape: [Weight,Height,Channel] ==> [1,Weight,Height,Channel]
        img_lr = tf.expand_dims(img_lr,axis=0)
        # Image Reconstructed 
        img_sr = self.model.predict(img_lr)
        # Reshape: [1,Weight,Height,Channel] -> [Weight,Height,Channel]
        img_sr = tf.squeeze(img_sr,axis=0)
        return img_sr
    def Plot(self,obj,head):
        """
        This function use the matplotlib.pyplot to plot the result
        Args:
            obj: Tensor type, the object which will be plotted
            head: string type, the title of the plotted image
        """
        fig = plt.figure()
        plt.imshow(obj)
        plt.title(head)
    def Plot_the_Result(self):
        """
        This function use to plot three result: Low-resolution, Original and Super-resolution image
        Args:
            img_lr: The low resolution image
            img_or: The original image
            img_sr: The reconstructed super-resolution image
        """
        num = np.random.randint(self.number)
        img_lr = self.test[num]
        img_or = self.train[num]
        img_sr = self.Image_SR(num)
        
        self.Plot(img_lr,'Low-Resolution Image')
        self.Plot(img_or,'Ground-True Image')
        self.Plot(img_sr,'Super-Resolution Image')
class Model_Design():
    def __init__(self,Variables):
        """
        Model Design class contain the Single Image Super-Resolution model (Generator),
        and the Classifier model (Discriminator).
        Args:
            channel: the image channel of the input image
            lr_size: the image size of the low-resolution image
            hr_size: the image size of the original image and reconstructed super-resolution image
        """
        self.channel = Variables.channel
        self.lr_size = Variables.lr_size
        self.hr_size = Variables.hr_size
        
        self.activation = 'relu'
    def Conv_Layer(self,inputs,filter_num=64,ks1=3,ks2=3,s1=1,s2=1,
                   Transpose=False,use_bias=False,only_conv=False,using_BN=False):
        # Convolution Layer in the Tensorflow
        """
        This is the Simple Convolution Block with BN Layer and Activation Layer
        Args:
            inputs: Tensor Type, the inputs of the Convolution Layer
            filter_num: Int Type, the filter size of the Convolution
            ks1,ks2: Int Type, the kernel size of the Convolution (ks1,ks2)
            s1,s2: Int Type, the strides of the Convolution (s1,s2)
            Transpose: Bool Type, if True, this is the Transpose Convolution; if False, this is the normal Convolution
            use_bias: Bool Type, decide the Convolution using the bias or not
            only_conv: Bool Type, decide the return object only convolution layer
        """
        initializer = tf.random_normal_initializer(0., 0.02)
        if Transpose:
            x1 = tf.keras.layers.Conv2DTranspose(filters=filter_num,kernel_size=(ks1,ks2),strides=(s1,s2),
                                                padding='same',use_bias=use_bias,kernel_initializer=initializer)(inputs)
        else:
            x1 = tf.keras.layers.Conv2D(filters=filter_num,kernel_size=(ks1,ks2),strides=(s1,s2),
                                       padding='same',use_bias=use_bias,kernel_initializer=initializer)(inputs)
        if only_conv:
            return x1
        else:
            if using_BN: # Batchnormalization Layer
                x2 = tf.keras.layers.BatchNormalization()(x1)
                return self.Activation(x2)
            else:
                return self.Activation(x1)
    def Activation(self,inputs):
        # ReLU
        """ Activation Function, which can create the non-linear model"""
        return tf.keras.layers.Activation('relu')(inputs)
    def ADD(self,inputs,shortcut):
        """ Add layer, can merge the feature between two neural layers"""
        return tf.keras.layers.add([inputs,shortcut])
    def Residual_Dense_Block(self,inputs,filter_num=64):
        """
        This Function aims to achieve the Residual Dense Block
        """
        # Golbal Residual Connection
        shortcut_0 = inputs
        # Layer - 1
        x = self.Conv_Layer(inputs)
        x = self.ADD(x,shortcut_0)
        shortcut_1 = x
        # Layer - 2
        x = self.Conv_Layer(x)
        x = self.ADD(x,shortcut_0)
        x = self.ADD(x,shortcut_1)
        shortcut_2 = x
        # Layer - 3
        x = self.Conv_Layer(x)
        x = self.ADD(x,shortcut_0)
        x = self.ADD(x,shortcut_1)
        x = self.ADD(x,shortcut_2)
        shortcut_3 = x
        # Layer - 4
        x = self.Conv_Layer(x)
        x = self.ADD(x,shortcut_0)
        x = self.ADD(x,shortcut_1)
        x = self.ADD(x,shortcut_2)
        x = self.ADD(x,shortcut_3)
        return x
    def Super_Resolution_Model(self):
        """
        Single Image Super Resolution model, which can upscale the model in 4x 
        Moreover, using the Residual Dense Block, which can share the feature between the layer,
        and overcome the gradient diffusion. Moreover, it can remove the Batchnormalization Layer and pooling layer
        Furthermore, the residual connection can improve the training efficiency of the model.
        """
        Inputs = tf.keras.Input(shape=(self.lr_size,self.lr_size,self.channel)) # Input Shape: [None,128,128,3]
        # Input Block
        x = self.Conv_Layer(Inputs,ks1=7,ks2=7)         # Shape: [None,256,256,64]
        x = self.Conv_Layer(x)                          # Shape: [None,256,256,64]
        shortcut_1 = x
        # First Iterative Block
        x = self.Residual_Dense_Block(x)                # Shape: [None,256,256,64]
        x = self.Residual_Dense_Block(x)                # Shape: [None,256,256,64]
        x = tf.keras.layers.BatchNormalization()(x)
        shortcut_2 = x
        x = self.Residual_Dense_Block(x)                # Shape: [None,256,256,64]
        x = self.Residual_Dense_Block(x)                # Shape: [None,256,256,64]
        x = tf.keras.layers.BatchNormalization()(x)
        shortcut_3 = x
        # Second Iterative Block
        x = self.Residual_Dense_Block(x)                # Shape: [None,256,256,64]
        x = self.Residual_Dense_Block(x)                # Shape: [None,256,256,64]
        x = tf.keras.layers.BatchNormalization()(x)
        shortcut_4 = x
        x = self.Residual_Dense_Block(x)                # Shape: [None,256,256,64]
        x = self.Residual_Dense_Block(x)                # Shape: [None,256,256,64]
        x = tf.keras.layers.BatchNormalization()(x)
        # Add Layer
        x = self.ADD(x,shortcut_1)
        x = self.ADD(x,shortcut_2)
        x = self.ADD(x,shortcut_3)
        x = self.ADD(x,shortcut_4)
        # Output Layer
        x = self.Conv_Layer(x)                          # Shape: [None,256,256,64]
        x = self.Conv_Layer(x)                          # Shape: [None,256,256,64]
        x = self.ADD(x,shortcut_1)                      # Shape: [None,256,256,64]
        x = tf.keras.layers.Conv2D(filters=3,kernel_size=(1,1),strides=(1,1),padding='same')(x) # Shape: 256,256,3
        
        # Create Model:
        sisr_model = tf.keras.models.Model(inputs=Inputs,outputs=x,name='Reconstruct_Model')
        sisr_model.summary()
        return sisr_model
class Loss_and_Optimizer():
    def __init__(self,vgg_19):
        """
        This class include the optimizer of the Generator and Optimizer.
        Moreover, the percentage of each Loss function is defined
        """
        self.Generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        
        self.MAE_Loss_Percentage = 1e-3 #0.001
        self.MSE_Loss_Percentage = 1.0
        self.VGG_Loss_Percentage = 1e-2 #0.001
        self.SSIM_Loss_Percentage = 1e-2 #0.001
        
        self.VGG_19 = vgg_19
    def Loss_of_Compile(self,y_true,y_pred):
        MSE_loss = tf.keras.losses.MSE(y_true=y_true,y_pred=y_pred)*self.MSE_Loss_Percentage
        MAE_loss = tf.keras.losses.MAE(y_true=y_true,y_pred=y_pred)*self.MAE_Loss_Percentage
        SSIM_loss = tf.reduce_mean(tf.image.ssim(img1=y_true,img2=y_pred,max_val=1))
        SSIM_loss = (1.0-SSIM_loss)*self.SSIM_Loss_Percentage
        Pixel_loss = MSE_loss + MAE_loss + SSIM_loss
        
        VGG_loss = tf.reduce_mean(tf.square(self.VGG_19(y_true) - self.VGG_19(y_pred)))    
        VGG_loss = VGG_loss*self.VGG_Loss_Percentage
        
        return (Pixel_loss+VGG_loss)
    
# Obtain the Basic Parameters
variables_class = Variables()

# Build the Model
model_class = Model_Design(variables_class)
Generator = model_class.Super_Resolution_Model()

# If you want to continue your training base on the trained model, please use below code
# Trained_Generator = tf.keras.models.load_model('Reconstructed_256_Ver1',compile=False)

# Build the VGG-19 Model, for Content Loss 
VGG_19 = tf.keras.applications.VGG19(include_top=False,
                                     input_shape=(variables_class.hr_size,variables_class.hr_size,variables_class.channel))
VGG = tf.keras.Model(inputs=VGG_19.input,
                     outputs=VGG_19.get_layer('block5_conv4').output)

# Obtain the Optimizer and Loss
Loss_Optimizer_class = Loss_and_Optimizer(VGG)

# Create the Tensor Datasets
y_true = Jpg_Tensor(file_path = variables_class.datasets_filepath,
                     Image_Number = variables_class.number_of_datasets,
                     ishape = variables_class.lr_size,
                     Channel = variables_class.channel).return_tensor()
x_train = Jpg_Tensor(file_path = variables_class.testing_datasets,
                     Image_Number = variables_class.number_of_datasets,
                     ishape = variables_class.hr_size,
                     Channel = variables_class.channel).return_tensor()
# Testing Datasets of the Trained Model
test = x_train

class Train():
    def __init__(self,variables,loss_and_optimizer,
                 x,y):
        self.Epochs = variables.epoch
        self.Batch_Size = variables.batch_size
        
        self.x_train = x[6:68]
        self.y_train = y[6:68] #  Slice function of the list 
        # Normally, use 20% for validation and 80% for training.
        self.x_test = x[0:5]
        self.y_test = y[0:5]
        self.validate = (self.x_test,self.y_test)
        
        self.loss_and_optimizer = loss_and_optimizer
    def train_init_Generator(self):
        Generator.compile(optimizer=self.loss_and_optimizer.Generator_optimizer,
                          loss=self.loss_and_optimizer.Loss_of_Compile,
                          metrics=['mae','mse'])
        Generator.fit(x=self.x_train,y=self.y_train,
                      validation_data=self.validate,
                      batch_size=self.Batch_Size,epochs=self.Epochs,
                      callbacks=[history])
    def train_on_existed(self,Model):
        SISR_Model = Model
        SISR_Model.compile(optimizer='adam',
                           loss=self.loss_and_optimizer.Loss_of_Compile,
                           metrics=['mae','mse'])
        SISR_Model.fit(x=self.x_train,y=self.y_train,
                      validation_data=(self.x_test,self.y_test),
                      batch_size=self.Batch_Size,epochs=self.Epochs,
                      callbacks=[history])

Train_class = Train(variables=variables_class,
                    loss_and_optimizer=Loss_Optimizer_class,
                    x=x_train,y=y_true)

Train_class.train_init_Generator()
#Train_class.train_on_existed(Model=Generator)
plot_the_result = Plot(test_datasets=test,
                       original_datasets=y_true,
                       model=Generator,
                       number_of_image=variables_class.number_of_datasets).Plot_the_Result()

with open('log.txt','a',encoding='utf-8') as f:
    for i in history.losses:
        f.write(str(i))
        f.write('\r')












































