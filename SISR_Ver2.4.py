# -*- coding: utf-8 -*-
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import pathlib
import datetime

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
            lr_size: The Image Size of the Low-Resolution Image
            mr_size: The Middle Image Size of the High-Resolution Image: 256*256
            hr_size: The Image Size of the High-Resolution Image and Super-Resolution Image
            channel: The Image Channel of the Image
            epoch: The epoch of the training
            batch_size: The batch size in each epochs (if the GPU high performance, the batch size should larger)
            number_of_datasets: The amount of the training datasets
            datasets_filepath: The filepath of the training datasets
            testing_datasets: The filepath of the testing datasets, use to validate the model performance
            using_cache_tensor: Using the cache can be faster, but need more Memory
        """
        self.lr_size = 128
        self.mi_size = 256
        self.hr_size = 512
        self.channel = 3
        self.epoch = 2
        self.batch_size = 1
        self.number_of_datasets = 15
        self.datasets_filepath = pathlib.Path(r'C:\Users\Jinpeng Liao\Desktop\Image Processing\512_JPG')
        self.datasets_LR = pathlib.Path(r'C:\Users\Jinpeng Liao\Desktop\Image Processing\128_JPG_Train')
        self.testing_datasets = pathlib.Path(r'C:\Users\Jinpeng Liao\Desktop\Image Processing\512_JPG_Test')
        self.using_cache_tensor = False
class Jpg_Tensor():
    def __init__(self, file_path, using_cache=False, Image_Number=10,ishape=64,Channel=3):
        """ 
        This Class has Build in Function(BIF) to process the Original Image to Tensor Type
        The Tensor type datasets can be the input datasets of the Network training
        Args:
            file_path: strings type, the file path of the image
            using_cache: Bool type, using the cache can be faster, but need more Memory
            Image_Number: Int type, It means the number of images to create the Tensor
            ishape: Int type, the output image shape will be ishape*ishape
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
        self.using_cache = using_cache
        
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
                   Transpose=False,use_bias=False,only_conv=False):
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
            return self.Activation(x1)
    def Downsampling(self,inputs,filter_num=64,s=2):
        """
        DownSample Residual Block
        Args:
            inputs: Tensor Type, the inputs of the Convolution Layer
            filter_num: Int Type, the filter size of the Convolution
            s: Int Type, the strides of the Conv_Layer. 
               It should be 1 or 2. If s=1, this is Idenitity Connection; if s=2, this is Convolution Connection
        """
        shortcut = inputs
        x = self.Conv_Layer(inputs=inputs,filter_num=filter_num,ks1=1,ks2=1,s1=1,s2=1)
        x = self.Conv_Layer(inputs=x, filter_num=filter_num,s1=s,s2=s)
        if s!= 1:
            shortcut = self.Conv_Layer(inputs=shortcut,filter_num=filter_num,s1=s,s2=s,
                                       only_conv=True)
        output = self.ADD(x,shortcut)
        output = self.Activation(output)
        return output
    def Upsampling(self,inputs,filter_num=64,s=2):
        x = self.Conv_Layer(inputs=inputs,filter_num=filter_num,ks1=1,ks2=1,s1=1,s2=1,
                            Transpose=True)
        x = self.Conv_Layer(inputs=x, filter_num=filter_num,s1=s,s2=s,
                            Transpose=True)        
        shortcut = self.Conv_Layer(inputs=inputs, filter_num=filter_num,s1=s,s2=s,
                                   Transpose=True,only_conv=True)
        output = self.ADD(x,shortcut)
        output = self.Activation(output)
        return output
    def Activation(self,inputs):
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

        return x
    def Iterative_Up_and_Down_Block(self,inputs,filter_num=64):
        x = self.Upsampling(inputs=inputs,filter_num=filter_num)
        x = self.Downsampling(inputs=x,filter_num=filter_num)
        return x
    def Classifiar(self):
        """
        The Classifiar Model can classify the Generated LR image and Original LR image
        Using the Maxpooling layer to reduce the model trainable parameters,
        Then, using the convolution layer and dense layer to create a simple image classification model
        """
        Inputs = tf.keras.Input(shape=(self.hr_size,self.hr_size,self.channel))
        
        x = self.Conv_Layer(inputs=Inputs)
        x = tf.keras.layers.MaxPool2D()(x)
        x = self.Conv_Layer(inputs=x)
        x = tf.keras.layers.MaxPool2D()(x)
        x = self.Conv_Layer(inputs=x)
        x = tf.keras.layers.MaxPool2D()(x)
        x = self.Conv_Layer(inputs=x)
        x = tf.keras.layers.MaxPool2D()(x)
        x = self.Conv_Layer(inputs=x)
        x = tf.keras.layers.MaxPool2D()(x)
        x = self.Conv_Layer(inputs=x)
        x = tf.keras.layers.MaxPool2D()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(32,activation=self.activation)(x)
        x = tf.keras.layers.Dense(1)(x)
        
        model = tf.keras.models.Model(inputs=Inputs,outputs=x,name='Classifier_Model')
        model.summary()
        return model  
    def Upsamle_2D(self,inputs):
        x = tf.keras.layers.UpSampling2D(size=(2,2),interpolation='bilinear')(inputs)
        x = self.Conv_Layer(x)
        x = self.Conv_Layer(x)
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
        x = self.Conv_Layer(Inputs,ks1=7,ks2=7,
                            only_conv=True)         # Shape: [None,128,128,64]
        # Shortcut 1
        shortcut_1 = x
        shortcut_1 = self.Upsamle_2D(shortcut_1)
        shortcut_1 = self.Upsamle_2D(shortcut_1)
        x = self.Conv_Layer(x,ks1=7,ks2=7)              # Shape: [None,128,128,64]

        # First Dense Block
        x = self.Residual_Dense_Block(x)                # Shape: [None,128,128,64]
        x = self.Residual_Dense_Block(x)                # Shape: [None,128,128,64]
        x = tf.keras.layers.BatchNormalization()(x)
        # Shortcut 2
        shortcut_2 = x
        shortcut_2 = self.Upsamle_2D(shortcut_2)
        shortcut_2 = self.Upsamle_2D(shortcut_2)
        # Second Dense Block
        x = self.Residual_Dense_Block(x)                # Shape: [None,256,256,64]
        x = self.Residual_Dense_Block(x)                # Shape: [None,256,256,64]
        x = tf.keras.layers.BatchNormalization()(x)
        # Shortcut 3
        shortcut_3 = x
        shortcut_3 = self.Upsamle_2D(shortcut_3)
        shortcut_3 = self.Upsamle_2D(shortcut_3)
        # Third Dense Block
        x = self.Residual_Dense_Block(x)                # Shape: [None,512,512,64]
        x = self.Residual_Dense_Block(x)                # Shape: [None,512,512,64]
        x = tf.keras.layers.BatchNormalization()(x)
        
        # Upsample Block
        x = self.Upsamle_2D(x)
        x = self.Upsamle_2D(x)
        # Add Layer
        x = self.ADD(x,shortcut_1)
        x = self.ADD(x,shortcut_2)
        x = self.ADD(x,shortcut_3)
        # Output Layer
        x = self.Conv_Layer(x,ks1=5,ks2=5)                          # Shape: [None,512,512,64]
        x = self.Conv_Layer(x)                          # Shape: [None,512,512,64]
        x = tf.keras.layers.Conv2D(filters=3,kernel_size=(1,1),strides=(1,1),padding='same')(x) # Shape: [None,512,512,64] => [None,512,512,3]
        
        # Create Model:
        sisr_model = tf.keras.models.Model(inputs=Inputs,outputs=x,name='Generator_SR_V2')
        sisr_model.summary()
        return sisr_model
class Loss_and_Optimizer():
    def __init__(self,vgg_19):
        """
        This class include the optimizer of the Generator and Optimizer.
        Moreover, the percentage of each Loss function is defined
        """
        self.Generator_optimizer_init = tf.keras.optimizers.Adam(5e-4)
        self.Generator_optimizer = tf.keras.optimizers.Adam(5e-4)
        self.Discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        
        self.Gen_Loss_Percentage = 1e-2
        self.MAE_Loss_Percentage = 5e-3
        self.MSE_Loss_Percentage = 1.0
        self.VGG_Loss_Percentage = 1e-3
        self.SSIM_Loss_Percentage = 5e-3
        
        self.VGG_19 = vgg_19
        self.VGG1 = tf.keras.Model(inputs=VGG_19.input,
                     outputs=VGG_19.get_layer('block5_conv4').output)
        self.VGG2 = tf.keras.Model(inputs=VGG_19.input,
                     outputs=VGG_19.get_layer('block2_conv2').output)
        self.VGG3 = tf.keras.Model(inputs=VGG_19.input,
                     outputs=VGG_19.get_layer('block1_conv2').output)
    def Loss_of_Discriminator(self,y_true,y_pred):
        """
        In this part, to reduce the confident of the Discriminator,
        the loss of the Discriminator should use the loss function like WGAN.
        args:
            y_true: the processed result of the Original Image from Discriminator.
            y_pred: the processed result of the Super-Resolution Image from Discriminator
        return:
            The Discriminator Loss
        """
        labels_09 = tf.ones_like(y_true) - 0.1
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_09,
                                                                       logits=y_true))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(y_pred),
                                                                           logits=y_pred))
        total_loss = real_loss+fake_loss
        return total_loss
    def Loss_of_Compile(self,y_true,y_pred):
        MSE_loss = tf.keras.losses.MSE(y_true=y_true,y_pred=y_pred)*self.MSE_Loss_Percentage
        MAE_loss = tf.keras.losses.MAE(y_true=y_true,y_pred=y_pred)*self.MAE_Loss_Percentage
        SSIM_loss = tf.reduce_mean(tf.image.ssim(img1=y_true,img2=y_pred,max_val=1))
        SSIM_loss = (1.0-SSIM_loss)*self.SSIM_Loss_Percentage
        Pixel_loss = MSE_loss + MAE_loss+SSIM_loss
        
        VGG1 = tf.reduce_mean(tf.square(self.VGG1(y_true) - self.VGG1(y_pred)))    
        VGG2 = tf.reduce_mean(tf.square(self.VGG2(y_true) - self.VGG2(y_pred)))   
        VGG3 = tf.reduce_mean(tf.square(self.VGG3(y_true) - self.VGG3(y_pred)))   
        VGG_loss = (VGG1+VGG2+VGG3)*self.VGG_Loss_Percentage
        
        return (Pixel_loss+VGG_loss)
    def Loss_of_Generator(self,y_true,y_pred,
                          SR_Image_512,GT_Image_512,SR_Image_256,GT_Image_256,
                          VGG_Real,VGG_Fake):
        """
        This function calculate the total loss of the generator (SISR Model),
        it includes the Adversarial Loss, Pixel Loss and VGG-19 Content Loss.
        """
        fake_logit = y_pred-tf.reduce_mean(y_true)
        real_logit = y_true-tf.reduce_mean(y_pred)
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(y_true),
                                                                           logits=real_logit))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(y_pred),
                                                                           logits=fake_logit))
        G_loss = (real_loss + fake_loss)*self.Gen_Loss_Percentage

        MSE_loss = tf.keras.losses.MSE(y_true=GT_Image_512,y_pred=SR_Image_512)*self.MSE_Loss_Percentage
        MAE_loss = tf.keras.losses.MAE(y_true=GT_Image_512,y_pred=SR_Image_512)*self.MAE_Loss_Percentage
        Pixel_loss = MSE_loss + MAE_loss

        VGG_loss = tf.reduce_mean(tf.square(VGG_Real - VGG_Fake))    
        VGG_loss = VGG_loss*self.VGG_Loss_Percentage
        
        Total_Loss = G_loss+Pixel_loss+VGG_loss
        return Total_Loss    
# Obtain the Basic Parameters
variables_class = Variables()

# Build the Model
model_class = Model_Design(variables_class)
#Discriminator = model_class.Classifiar()
Generator = model_class.Super_Resolution_Model()
#Trained_Generator = tf.keras.models.load_model('SISR_Ver2_4',compile=False)
# Build the VGG-19 Content Loss Model
VGG_19 = tf.keras.applications.VGG19(include_top=False,
                                     input_shape=(variables_class.hr_size,variables_class.hr_size,variables_class.channel))
VGG = tf.keras.Model(inputs=VGG_19.input,
                     outputs=VGG_19.get_layer('block5_conv4').output)
# Obtain the Optimizer and Loss
Loss_Optimizer_class = Loss_and_Optimizer(VGG_19)
# Obtain the Tensor Datasets
y_train = Jpg_Tensor(file_path = variables_class.datasets_filepath,
                     using_cache = variables_class.using_cache_tensor,
                     Image_Number = variables_class.number_of_datasets,
                     ishape = variables_class.hr_size,
                     Channel = variables_class.channel).return_tensor()
x_train = Jpg_Tensor(file_path = variables_class.datasets_filepath,
                     using_cache = variables_class.using_cache_tensor,
                     Image_Number = variables_class.number_of_datasets,
                     ishape = variables_class.lr_size,
                     Channel = variables_class.channel).return_tensor()
# Training Datasets for WGAN
y = tf.data.Dataset.from_tensor_slices(y_train).batch(variables_class.batch_size)
x = tf.data.Dataset.from_tensor_slices(x_train).batch(variables_class.batch_size)
# Testing Datasets of the Trained Model
test = x_train

class Train():
    def __init__(self,variables,loss_and_optimizer,
                 x_train_com,y_train_com,x_train_gan,y_train_gan):
        self.Epochs = variables.epoch
        self.Batch_Size = variables.batch_size
        
        self.x_train = x_train_com
        self.y_train = y_train_com
        
        self.x_train_gan = x_train_gan
        self.y_train_gan = y_train_gan
        
        self.loss_and_optimizer = loss_and_optimizer
    def loss_fun(self,y_true, y_pred):
        loss = 0.001*tf.keras.losses.MAE(y_true=y_true,y_pred=y_pred)+tf.keras.losses.MSE(y_true=y_true,y_pred=y_pred)
        return loss
    def train_init_Generator(self):
        Generator.compile(optimizer=self.loss_and_optimizer.Generator_optimizer,
                          loss=self.loss_and_optimizer.Loss_of_Compile,
                          metrics=['mse'])
        Generator.fit(x=self.x_train,y=self.y_train,
                      batch_size=self.Batch_Size,epochs=self.Epochs,
                      callbacks=[history])
    def train_on_existed(self,Model):
        SISR_Model = Model
        SISR_Model.compile(optimizer=self.loss_and_optimizer.Generator_optimizer,
                           loss=self.loss_and_optimizer.Loss_of_Compile,
                           metrics=['mae','mse'])
        SISR_Model.fit(x=self.x_train,y=self.y_train,
                      batch_size=self.Batch_Size,epochs=self.Epochs,
                      callbacks=[history])
    def training(self):
        # Init Gen Training
        print("*"*60)
        print("Training the Init Generator")
        self.train_init_Generator()
        print("Finish Training the Init Generator, Now Training the GAN")
        print("*"*60)
        
        # GAN Training
        epoch_loss_avg_g = tf.keras.metrics.Mean()
        epoch_loss_avg_d = tf.keras.metrics.Mean() 
        epoch_accuracy_d = tf.keras.metrics.BinaryAccuracy() 
        Epoch_for_GAN = int(self.Epochs/4)
        for epoch in range(Epoch_for_GAN):
            print('Training GAN at Epoch: {}/{}'.format(epoch+1,Epoch_for_GAN))
            for L,S in zip(self.x_train_gan,self.y_train_gan):
                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    Fake_Image = Generator(L,training=True) # Output = [None,256,256,3]
            
                    real_output = Discriminator(S,training=True) # Output = [None,1]
                    fake_output = Discriminator(Fake_Image,training=True) # Output = [None,1]
                    
                    Content_Real = VGG(S)
                    Content_Fake = VGG(Fake_Image)
                    
                    gen_loss = self.loss_and_optimizer.Loss_of_Generator(real_output,fake_output,
                                           SR_Image_512=Fake_Image,GT_Image_512=S,
                                           SR_Image_256=0,GT_Image_256=0,
                                           VGG_Real=Content_Real ,VGG_Fake=Content_Fake)
                    
                    disc_loss = self.loss_and_optimizer.Loss_of_Discriminator(real_output,fake_output)
                    
                    epoch_loss_avg_g(gen_loss) 
                    epoch_loss_avg_d(disc_loss)
                    
                    epoch_accuracy_d.update_state(y_true=tf.ones_like(real_output),y_pred=tf.ones_like(fake_output))
                    
                gradients_of_generator = gen_tape.gradient(gen_loss, Generator.trainable_variables)
                gradients_of_discriminator = disc_tape.gradient(disc_loss, Discriminator.trainable_variables)
    
                self.loss_and_optimizer.Generator_optimizer.apply_gradients(zip(gradients_of_generator, Generator.trainable_variables))
                self.loss_and_optimizer.Discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, Discriminator.trainable_variables))
        
            if epoch %1 ==0:
                print("=="*40)
                print("Epoch {:03d}: G_Loss: {:.7f}, D_Loss:{:.7f}, D_ACC: {:.7f}%".format(epoch+1,
                                                                                      epoch_loss_avg_g.result(),
                                                                                      epoch_loss_avg_d.result(),
                                                                                      100*epoch_accuracy_d.result().numpy()))
            print("=="*40)

Train_class = Train(variables=variables_class,loss_and_optimizer=Loss_Optimizer_class,
                    x_train_com=x_train,y_train_com=y_train,x_train_gan=x,y_train_gan=y)

#Train_class.train_on_existed(Trained_Generator)
Train_class.train_init_Generator()
#Train_class.train_init_Generator()
plot_the_result = Plot(test_datasets=test,
                       original_datasets=y_train,
                       model=Trained_Generator,
                       number_of_image=variables_class.number_of_datasets).Plot_the_Result()
with open('log_4.txt','a',encoding='utf-8') as f:
    for i in history.losses:
        f.write(str(i))
        f.write('\r')
        
        
        
        
        
        