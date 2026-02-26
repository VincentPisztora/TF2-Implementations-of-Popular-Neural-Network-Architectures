# -*- coding: utf-8 -*-

###############################################################################

#References:
#https://github.com/jimmyyhwu/resnet18-tf2/blob/master/resnet.py
#https://github.com/pytorch/vision/blob/v0.4.0/torchvision/models/resnet.py

###############################################################################

from tensorflow.keras.layers import Lambda, Dense
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, BatchNormalization, Add, ReLU, GlobalAveragePooling2D

import tensorflow as tf
from tensorflow import keras
import keras as keras_2

###############################################################################

@keras.saving.register_keras_serializable()
class GenericClassificationHead(keras.layers.Layer):
    def __init__(self,n_classes,dense_dim,dropout,weight_decay,ln_epsilon=1e-6):
        super(GenericClassificationHead,self).__init__()
        self.n_classes = n_classes
        self.dense_dim = dense_dim
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.ln_epsilon = ln_epsilon
        
        self.d_1 = Dense(self.dense_dim,activation=tf.keras.activations.relu)
        self.d_2 = Dense(self.n_classes)
        
    def call(self,inputs,training=False): #[batch_size,embed_dim]
        z = self.d_1(inputs) #[batch_size,dense_dim]
        z = self.d_2(z) #[batch_size,n_classes]
        return z
    
    def get_config(self):
        config = {'dense_dim':self.dense_dim,
                  'dropout':self.dropout,
                  'n_classes':self.n_classes,
                  'weight_decay':self.weight_decay,
                  'ln_epsilon':self.ln_epsilon}
        return config
    
    @classmethod
    def from_config(cls,config):
        return cls(**config)
    
@keras_2.saving.register_keras_serializable()
class BasicBlock(keras.Model):
    def __init__(self, planes, stride=1, downsample=None, name=None, **kwargs):
        super(BasicBlock,self).__init__(**kwargs)
        self.planes = planes
        self.stride = stride
        self.downsample = downsample
        self.name = name
        
        self.zp_1 = ZeroPadding2D(padding=1)
        self.conv_1 = Conv2D(filters=self.planes, kernel_size=3, strides=self.stride, use_bias=False, kernel_initializer=keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='untruncated_normal'))
        self.bn_1 = BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.a_1 = ReLU()
        
        self.zp_2 = ZeroPadding2D(padding=1)
        self.conv_2 = Conv2D(filters=self.planes, kernel_size=3, strides=1, use_bias=False, kernel_initializer=keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='untruncated_normal'))
        self.bn_2 = BatchNormalization(momentum=0.9, epsilon=1e-5)
        
        if self.downsample is not None:
            self.conv_3 = Conv2D(filters=self.downsample, kernel_size=1, strides=self.stride, use_bias=False, kernel_initializer=keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='untruncated_normal'))
            self.bn_3 = BatchNormalization(momentum=0.9, epsilon=1e-5)
            
            self.downsample = [self.conv_3,self.bn_3]
        else:
            self.downsample = []
        
        self.add_1 = Add()
        self.a_2 = ReLU()
        
        
    def call(self,inputs,training=False):
        identity = inputs
        
        z = self.zp_1(inputs)
        z = self.conv_1(z)
        z = self.bn_1(z,training=training)
        z = self.a_1(z)
        
        z = self.zp_2(z)
        z = self.conv_2(z)
        z = self.bn_2(z,training=training)
        
        for downsample_layer in self.downsample:
            identity = downsample_layer(identity,training=training)
        
        z = self.add_1([identity,z])
        z = self.a_2(z)
        return z
    
    def get_config(self):
        config = {}
        config.update({'planes':self.planes,'stride':self.stride,'downsample':self.downsample})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

@keras_2.saving.register_keras_serializable()
class BigBlock(keras.Model):
    def __init__(self, inplanes, planes, blocks, stride=1, name=None, **kwargs):
        super(BigBlock,self).__init__(**kwargs)
        self.inplanes = inplanes
        self.planes = planes
        self.blocks = blocks
        self.stride = stride
        self.name = name
        
        self.downsample = None
        if self.stride != 1 or self.inplanes != self.planes:
            self.downsample = self.planes
        
        self.bb_1 = BasicBlock(self.planes, self.stride, self.downsample, name=f'{name}.0')
        self.bbs = []
        for i in range(1, self.blocks):
            self.bbs.append(BasicBlock(self.planes, name=f'{name}.{i}'))
        
    def call(self,inputs,training=False):
        z = self.bb_1(inputs,training=training)
        for i in range(0, self.blocks-1):
            z = self.bbs[i](z,training=training)
        return z
    
    def get_config(self):
        config = {}
        config.update({'inplanes':self.inplanes,'planes':self.planes,'blocks':self.blocks,'stride':self.stride,'name':self.name})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

@keras_2.saving.register_keras_serializable()
class R18BaseModel(keras.Model):
    def __init__(self,**kwargs):
        super(R18BaseModel,self).__init__(**kwargs)
        self.zp_1 = ZeroPadding2D(padding=3, name='conv1_pad')
        self.conv_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer=keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='untruncated_normal'), name='conv1')
        self.bn_1 = BatchNormalization(momentum=0.9, epsilon=1e-5, name='bn1')
        self.a_1 = ReLU(name='relu1')
        
        blocks_per_layer = [2,2,2,2]
        
        self.bigb_1 = BigBlock(64, 64, blocks_per_layer[0], name='layer1')
        self.bigb_2 = BigBlock(64, 128, blocks_per_layer[1], stride=2, name='layer2')
        self.bigb_3 = BigBlock(128, 256, blocks_per_layer[2], stride=2, name='layer3')
        self.bigb_4 = BigBlock(256, 512, blocks_per_layer[3], stride=2, name='layer4')

        self.gap = GlobalAveragePooling2D(name='avgpool')
        
    def call(self,inputs,training=False):
        z = self.zp_1(inputs)
        z = self.conv_1(z)
        z = self.bn_1(z,training=training)
        z = self.a_1(z)
        
        z = self.bigb_1(z,training=training)
        z = self.bigb_2(z,training=training)
        z = self.bigb_3(z,training=training)
        z = self.bigb_4(z,training=training)
        
        z = self.gap(z)
        return z
    
    def get_config(self):
        config = {}
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

#out_dim: The dimension of the last Dense layer of the classification_head
#proj_dim: The dimension of the embedding layer used downstream
#n_classes: The dimensions of the linear_probe Dense layers
#n_regr: the number of regression probe tasks - each corresponding to one Dense(1) layer
class ResNet18(keras.Model):
    def __init__(self,output_size,proj_dim,with_top,dropout,weight_decay,n_classes,n_regr,**kwargs):
        super(ResNet18,self).__init__(**kwargs)
        self.with_top = with_top
        self.output_size = output_size
        self.proj_dim = proj_dim
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.n_classes = n_classes
        self.n_regr = n_regr
        
        self.base_model = R18BaseModel()
        
        if self.with_top:
            self.top_layer = GenericClassificationHead(n_classes=self.output_size,
                                                       dense_dim=self.proj_dim,
                                                       dropout=self.dropout,
                                                       weight_decay=self.weight_decay)
        else:
            self.top_layer = Lambda(lambda x: x)
            
        self.classifier_probes = []
        for n_class in self.n_classes:
            self.classifier_probes.append(Dense(n_class))
        
        self.regressor_probes = []
        for i in range(self.n_regr):
            self.regressor_probes.append(Dense(1))
                
    def call(self,inputs,training=False):
        z = self.base_model(inputs,training=training)
        logits = self.top_layer(z,training=training)
        classifier_preds = []
        for classifier in self.classifier_probes:
            classifier_preds.append(classifier(tf.stop_gradient(z)))
        regressor_preds = []
        for regressor in self.regressor_probes:
            regressor_preds.append(regressor(tf.stop_gradient(z)))
        return z,logits,classifier_preds,regressor_preds
    
    def get_config(self):
        config = {}
        config.update({'output_size':self.output_size,'with_top':self.with_top})
        config.update({'dropout':self.dropout,'weight_decay':self.weight_decay,'proj_dim':self.proj_dim})
        config.update({'n_classes':self.n_classes,'n_regr':self.n_regr})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

