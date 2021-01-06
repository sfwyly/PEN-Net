# 复现论文 Learning Pyramid-Context Encoder Network for High-Quality Image Inpainting
# CVPR 2019 微软亚洲研究院 zyh

import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import numpy as np

class AtnConv(layers.Layer):
    def __init__(self,input_channels = 128, output_channels = 64, groups = 4, ksize = 3, stride = 1, rate = 2,softmax_scale = 10,fuse = True, rates = [1,2,4,8]):
        super(AtnConv,self).__init__()
        
        self.kernel = ksize
        self.strides = stride
        self.rate = rate
        self.softmax_scale = softmax_scale
        self.groups = groups
        self.fuse = fuse
        
        if(self.fuse):
            self.group_blocks = []
            for i in range(groups):
                self.group_blocks.append(
                    models.Sequential([layers.Conv2D(output_channels//groups,kernel_size=3,dilation_rate=rates[i],padding="same"),layers.ReLU()])
                )
    #x1 lower-level  x2: high-level
    def call(self,x1,x2,mask  =None):

        x1s = x1.shape
        x2s = x2.shape
        kernel = 2*self.rate
        raw_w = tf.image.extract_patches(x1, [1,self.kernel,self.kernel,1], [1,self.rate*self.strides,self.rate*self.strides,1], [1,1,1,1], padding='SAME')
        raw_w = tf.reshape(raw_w, [x1s[0], -1, self.kernel, self.kernel, x1s[-1]]) 
        raw_w = tf.transpose(raw_w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
        raw_w_groups = tf.split(raw_w,x1s[0],axis = 0)

        
        f_groups  =tf.split(x2,x2s[0],axis= 0)
       
        w = tf.image.extract_patches(x2, [1,self.kernel,self.kernel,1], [1,self.strides,self.strides,1], [1,1,1,1], padding='SAME')
        w = tf.reshape(w, [x2s[0], -1, self.kernel, self.kernel, x2s[-1]]) 
        w = tf.transpose(w, [0, 2, 3, 4, 1])
        w_groups = tf.split(w,x2s[0],axis= 0)

        ms = mask.shape
        if(mask is not None):
            mask = tf.image.resize(mask,x2s[1:3],"bilinear")
        else:
            mask = tf.zeros((x2s[0],x2s[1],x2s[2],x2s[3]))
        m = tf.image.extract_patches(mask, [1,self.kernel,self.kernel,1], [1,self.strides,self.strides,1], [1,1,1,1], padding='SAME')
        m = tf.reshape(m, [ms[0], -1, self.kernel, self.kernel, ms[-1]]) 
        m = tf.transpose(m, [0, 2, 3, 4, 1])# b k k c hw
        m = tf.cast(tf.equal(tf.reduce_mean(m, axis=[1,2,3], keepdims=True), 1.), tf.float32)
        mm = tf.squeeze(m,axis = 1) #b 1 1 hw
        mm_groups = tf.split(mm,ms[0],axis= 0)

        y = []
        scale = self.softmax_scale
        for xi,wi,raw_wi,mi in zip(f_groups,w_groups,raw_w_groups,mm_groups):
            
            wi =wi[0] #k k c hw
            escape_NaN = 1e-4
            wi_normed = wi / tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(wi), axis=[0,1,2])), escape_NaN)#k k c hw
            
            yi = tf.nn.conv2d(xi, wi_normed, strides=1, padding="SAME")#1 h w  hw
            
            yi = tf.reshape(yi,[1,x2s[1],x2s[2],x2s[1]//self.strides*x2s[2]//self.strides])
            
            yi = yi * mi
            yi = tf.nn.softmax(yi*scale,axis = -1)
            yi = yi*mi
            yi = tf.clip_by_value(yi,10e-8,10e8)
            
            wi_center = raw_wi[0]
            yi = tf.nn.conv2d_transpose(yi,wi_center,tf.concat([[1], x1s[1:]], axis=0),strides=[1,self.rate,self.rate,1],padding="SAME")/4.
            y.append(yi)
        y = tf.concat(y,axis = 0)
        if(self.fuse):
            tmp = []
            for i in range(self.groups):
                tmp.append(self.group_blocks[i](y))
            y = tf.concat(tmp,axis= -1)
        
        return y

class InpaintGenerator(models.Model):
    
    def __init__(self):
        super(InpaintGenerator,self).__init__()
        
        cnum = 32
        
        self.dw_conv01 = models.Sequential([layers.Conv2D(cnum,kernel_size = 3,strides = 2,padding="same"),
                                          layers.LeakyReLU(0.2)])
        self.dw_conv02 = models.Sequential([layers.Conv2D(2*cnum,kernel_size = 3,strides = 2,padding="same"),
                                          layers.LeakyReLU(0.2)])
        self.dw_conv03 = models.Sequential([layers.Conv2D(4*cnum,kernel_size = 3,strides = 2,padding="same"),
                                          layers.LeakyReLU(0.2)])
        self.dw_conv04 = models.Sequential([layers.Conv2D(8*cnum,kernel_size = 3,strides = 2,padding="same"),
                                          layers.LeakyReLU(0.2)])
        self.dw_conv05 = models.Sequential([layers.Conv2D(16*cnum,kernel_size = 3,strides = 2,padding="same"),
                                          layers.LeakyReLU(0.2)])
        self.dw_conv06 = models.Sequential([layers.Conv2D(16*cnum,kernel_size = 3,strides = 2,padding="same"),
                                          layers.ReLU()])
        
        #attention module
        self.at_conv05 = AtnConv(cnum*16,cnum*16,ksize =  3,fuse = False)
        self.at_conv04 = AtnConv(cnum*8,cnum*8)
        self.at_conv03 = AtnConv(cnum*4,cnum*4)
        self.at_conv02 = AtnConv(cnum*2,cnum*2)
        self.at_conv01 = AtnConv(cnum,cnum)
        
        #decoder
        self.up_conv05 = models.Sequential([layers.Conv2D(16*cnum,kernel_size = 3,strides = 1,padding="same"),
                                           layers.ReLU()])
        self.up_conv04 = models.Sequential([layers.Conv2D(8*cnum,kernel_size = 3,strides = 1,padding="same"),
                                           layers.ReLU()])
        self.up_conv03 = models.Sequential([layers.Conv2D(4*cnum,kernel_size = 3,strides = 1,padding="same"),
                                           layers.ReLU()])
        self.up_conv02 = models.Sequential([layers.Conv2D(2*cnum,kernel_size = 3,strides = 1,padding="same"),
                                           layers.ReLU()])
        self.up_conv01 = models.Sequential([layers.Conv2D(cnum,kernel_size = 3,strides = 1,padding="same"),
                                           layers.ReLU()])
        
        # torgb
        self.torgb5 = models.Sequential([layers.Conv2D(3,kernel_size = 1,strides = 1,activation="tanh",padding="same")])
        self.torgb4 = models.Sequential([layers.Conv2D(3,kernel_size = 1,strides = 1,activation="tanh",padding="same")])
        self.torgb3 = models.Sequential([layers.Conv2D(3,kernel_size = 1,strides = 1,activation="tanh",padding="same")])
        self.torgb2 = models.Sequential([layers.Conv2D(3,kernel_size = 1,strides = 1,activation="tanh",padding="same")])
        self.torgb1 = models.Sequential([layers.Conv2D(3,kernel_size = 1,strides = 1,activation="tanh",padding="same")])
        
        self.decoder = models.Sequential([layers.Conv2D(cnum,kernel_size = 3,strides = 1,activation="relu",padding="same"),
                                         layers.Conv2D(3,kernel_size = 3,strides = 1,activation="tanh",padding="same")])
    def call(self,img,mask):
        
        #x = img
        x = tf.concat([img,mask],axis = -1)
        # encoder
        x1 = self.dw_conv01(x)
        x2 = self.dw_conv02(x1)
        x3 = self.dw_conv03(x2)
        x4 = self.dw_conv04(x3)
        x5 = self.dw_conv05(x4)
        x6 = self.dw_conv06(x5)
        
        # attention
        x5 = self.at_conv05(x5,x6,mask)
        x4 = self.at_conv04(x4,x5,mask)
        x3 = self.at_conv03(x3,x4,mask)
        x2 = self.at_conv02(x2,x3,mask)
        x1 = self.at_conv01(x1,x2,mask)
        
        # decoder
        upx5 = self.up_conv05(tf.image.resize(x6,(x6.shape[1]*2,x6.shape[2]*2),method = "bilinear"))
        upx4 = self.up_conv04(tf.image.resize(tf.concat([upx5,x5],axis = -1),(x5.shape[1]*2,x5.shape[2]*2),method = "bilinear"))
        upx3 = self.up_conv03(tf.image.resize(tf.concat([upx4,x4],axis = -1),(x4.shape[1]*2,x4.shape[2]*2),method = "bilinear"))
        upx2 = self.up_conv02(tf.image.resize(tf.concat([upx3,x3],axis = -1),(x3.shape[1]*2,x3.shape[2]*2),method = "bilinear"))
        upx1 = self.up_conv01(tf.image.resize(tf.concat([upx2,x2],axis = -1),(x2.shape[1]*2,x2.shape[2]*2),method = "bilinear"))
        
        # torgb
        img5 = self.torgb5(tf.concat([upx5,x5],axis = -1))
        img4 = self.torgb4(tf.concat([upx4,x4],axis = -1))
        img3 = self.torgb3(tf.concat([upx3,x3],axis = -1))
        img2 = self.torgb2(tf.concat([upx2,x2],axis = -1))
        img1 = self.torgb1(tf.concat([upx1,x1],axis = -1))
        
        # output
        output = self.decoder(tf.image.resize(tf.concat([upx1,x1],axis = -1),(x1.shape[1]*2,x1.shape[2]*2),method = "bilinear"))
        pyramid_imgs = [img1, img2, img3, img4, img5]
        
        return pyramid_imgs, output
    
class Discriminator(models.Model):
    
    def __init__(self,in_channels,use_sigmoid = False,use_sn = True,init_weights = True):
        super(Discriminator,self).__init__()
        self.use_sigmoid = use_sigmoid
        cnum = 64
        
        self.encoder = models.Sequential([
            
            layers.Conv2D(cnum,kernel_size = 5,strides = 2,padding = "same",use_bias = False),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            
            layers.Conv2D(2*cnum,kernel_size = 5,strides = 2,padding = "same",use_bias = False),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            
            layers.Conv2D(4*cnum,kernel_size = 5,strides = 2,padding = "same",use_bias = False),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            
            layers.Conv2D(8*cnum,kernel_size = 5,strides = 2,padding = "same",use_bias = False),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2)
        ])
        
        self.classifier = layers.Conv2D(1,kernel_size = 5,strides = 1,padding = "same")
        
    def call(self,x):
        x = self.encoder(x)
        labels_x = self.classifier(x)
        if(self.use_sigmoid):
            labels_x = tf.nn.sigmoid(labels_x)
        return labels_x
        
inputs = layers.Input(batch_shape = (6,256,256,3))
masks  = layers.Input(batch_shape = (6,256,256,3))
outputs = InpaintGenerator()(inputs,masks)
generator = models.Model(inputs = [inputs,masks],outputs = outputs)
generator.summary()
