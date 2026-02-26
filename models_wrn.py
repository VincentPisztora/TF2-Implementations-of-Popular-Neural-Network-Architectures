# -*- coding: utf-8 -*-

###############################################################################

#References:
#https://urldefense.com/v3/__https://rpmarchildon.com/wp-content/uploads/2018/09/RM-W-Keras-VGG-WRN-vF1.html*section_3__;Iw!!KVWo1iE!RdS6Q7nj7b4U_0RM76WDz-JXxp1iegsRk18mH2bHe-1Jg8OsAYUQwRIekCLBCHegva-96kXmEmHrUPd-ohXf$ 
#https://urldefense.com/v3/__https://arxiv.org/pdf/1605.07146.pdf__;!!KVWo1iE!RdS6Q7nj7b4U_0RM76WDz-JXxp1iegsRk18mH2bHe-1Jg8OsAYUQwRIekCLBCHegva-96kXmEmHrUPl4FysJ$ 
#https://urldefense.com/v3/__https://github.com/keras-team/keras/blob/v2.8.0/keras/layers/normalization/batch_normalization.py*L1125-L1265__;Iw!!KVWo1iE!RdS6Q7nj7b4U_0RM76WDz-JXxp1iegsRk18mH2bHe-1Jg8OsAYUQwRIekCLBCHegva-96kXmEmHrUNmMfQYI$ 

###############################################################################

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Flatten, Dense, Conv2D, BatchNormalization
from tensorflow.keras.layers import Lambda, ReLU, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.regularizers import L2

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

class WRNConv1Block(keras.Model):
    def __init__(self,filters,weight_decay,**kwargs):
        super(WRNConv1Block,self).__init__(**kwargs)
        self.filters = filters
        self.weight_decay = weight_decay
        self.conv = Conv2D(filters=self.filters,kernel_size=(3,3),strides=(1,1),padding='same',
                           kernel_regularizer=L2(self.weight_decay))
        
    def call(self,inputs):        
        z = self.conv(inputs)
        
        return z
    
    def get_config(self):
        config = {}
        config.update({'filters':self.filters,'weight_decay':self.weight_decay})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

#l 'deepening factor': num convolutions per block, best performance found in paper with l=2
#N: num blocks within each group [N = (n-4)/(3*l)]
class WRNGroup(keras.Model):
    def __init__(self,filters,N,drop_p,weight_decay,strides,**kwargs):
        super(WRNGroup,self).__init__(**kwargs)
                
        self.filters = filters
        self.N = N
        self.drop_p = drop_p
        self.weight_decay = weight_decay
        self.strides = strides
        
        self.blocks_dict = {}
        for i in range(self.N):
            self.blocks_dict.update({'WRN_Block_'+str(i):WRNMidBlock(block_i=i,
                                                                     filters=filters,drop_p=drop_p,
                                                                     strides=self.strides,
                                                                     weight_decay=self.weight_decay)})
        
    def call(self,inputs):
        for key,block in self.blocks_dict.items():
            if block.block_i == 0:
                z = block(inputs)
            else:
                z = block(z)
        return z
    
    def get_config(self):
        config = {}
        config.update({'filters':self.filters,'N':self.N,'drop_p':self.drop_p,'weight_decay':self.weight_decay,
                       'strides':self.strides})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class WRNMidBlock(keras.Model):
    def __init__(self,block_i,filters,drop_p,strides,weight_decay,**kwargs):
        super(WRNMidBlock,self).__init__(**kwargs)
        self.inputs_shape = None
        
        self.block_i = block_i
        self.filters = filters
        self.drop_p = drop_p
        self.strides = strides
        self.weight_decay = weight_decay
        
        if self.block_i == 0:
            self.res_conv = Conv2D(filters=filters,kernel_size=1,strides=self.strides,padding='same',kernel_regularizer=L2(self.weight_decay))
            self.conv0 = Conv2D(filters=filters,kernel_size=3,strides=self.strides,padding='same',kernel_regularizer=L2(self.weight_decay))
        else:
            self.res_conv = Lambda(lambda x: x)
            self.conv0 = Conv2D(filters=filters,kernel_size=3,strides=1,padding='same',kernel_regularizer=L2(self.weight_decay))
        
        self.bn0 = BatchNormalization()
        self.act0 = ReLU()
        self.conv1 = Conv2D(filters=filters,kernel_size=3,strides=1,padding='same',kernel_regularizer=L2(self.weight_decay))
        self.bn1 = BatchNormalization()
        self.act1 = ReLU()
        self.dropout = Dropout(rate=self.drop_p)
        self.add = tf.keras.layers.Add()
                
    def call(self,inputs):
        
        z0 = self.res_conv(inputs)
        
        z1 = self.bn0(inputs)
        z1 = self.act0(z1)
        
        z1 = self.conv0(z1)
        z1 = self.bn1(z1)
        z1 = self.act1(z1)
        z1 = self.dropout(z1)
        z1 = self.conv1(z1)
        
        z = self.add([z0, z1])
        
        return z
    
    def get_config(self):
        config = {}
        config.update({'block_i':self.block_i,'filters':self.filters,
                       'drop_p':self.drop_p,'strides':self.strides})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


#num_base_filters: num conv filters in conv1_block (gets multiplied later by k)
#k 'widening factor': multiplies the num of features in conv layers
#l 'deepening factor': num convolutions per block, best performance found in paper with l=2 which is what is used here
#N: num blocks within each group [N = (n-4)/(3*l)]
#n: num of convolutions in whole model [n = 1 + (1+l*N) + (1+l*N) + (1+l*N) -> n = 4 + 3*l*N]
#WRN naming convention is WRN-n-k
#So e.g. (n:28,k:10) == (N:4,k:10)
class WideResNet(keras.Model):
    def __init__(self,output_size,with_top,num_base_filters,N,k,drop_p,weight_decay,n_class,**kwargs):
        super(WideResNet,self).__init__(**kwargs)
        self.output_size = output_size
        self.with_top = with_top
        self.num_base_filters = num_base_filters
        self.N = N
        self.k = k
        self.drop_p = drop_p
        self.weight_decay = weight_decay
        self.n_class = n_class
        
        self.conv1_block = WRNConv1Block(filters=self.num_base_filters,weight_decay=self.weight_decay)
        self.groups_dict = {}
        for i in range(3):
            if i == 0:
                self.groups_dict.update({'WRN_Group_'+str(i):WRNGroup(filters=self.num_base_filters*(2**i)*self.k,
                                                                      N=self.N,drop_p=self.drop_p,
                                                                      weight_decay=self.weight_decay,strides=1)})
            else:
                self.groups_dict.update({'WRN_Group_'+str(i):WRNGroup(filters=self.num_base_filters*(2**i)*self.k,
                                                                      N=self.N,drop_p=self.drop_p,
                                                                      weight_decay=self.weight_decay,strides=2)})
        self.bn = BatchNormalization()
        self.act = ReLU()
        self.avg_pool = GlobalAveragePooling2D()
        self.flat = Flatten()
        
        
        
        self.embed_resize = Dense(self.output_size,activation=None,kernel_regularizer=L2(self.weight_decay))
        
        if self.with_top:
            self.top_layer = GenericClassificationHead(n_classes=self.output_size,
                                                       dense_dim=self.output_size,
                                                       dropout=self.dropout,
                                                       weight_decay=self.weight_decay)
        else:
            self.top_layer = Lambda(lambda x: x)
            
        self.linear_probe = Dense(self.n_class)
        
    def call(self,inputs,training=None):
        
        z = self.conv1_block(inputs)
        for key,group in self.groups_dict.items():
            z = group(z)
        z = self.bn(z)
        z = self.act(z)
        
        z = self.avg_pool(z)
        
        z = self.embed_resize(self.flat(z))
        
        logits = self.top_layer(z)
        
        logits_probe = self.linear_probe(tf.stop_gradient(z))
        
        return z,logits,logits_probe
    
    def get_config(self):
        config = {}
        config.update({'output_size':self.output_size,'with_top':self.with_top,
                       'num_base_filters':self.num_base_filters,'N':self.N,
                       'k':self.k,'drop_p':self.drop_p,'weight_decay':self.weight_decay,
                       'n_class':self.n_class})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
