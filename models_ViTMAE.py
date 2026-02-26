# -*- coding: utf-8 -*-

###############################################################################

#References:
#https://github.com/emla2805/vision-transformer/blob/master/model.py
#https://github.com/google-research/vision_transformer/blob/main/vit_jax/train.py
#https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention
#https://github.com/ariG23498/mae-scalable-vision-learners/blob/master/mae-pretraining.ipynb

###############################################################################

import tensorflow as tf
from tensorflow import keras
import keras as keras_2
from keras.layers import Dense, Dropout, LayerNormalization, GlobalAveragePooling1D, Reshape
from tensorflow.keras.regularizers import L2

###############################################################################

class PositionalEmbeddingLayer(keras.layers.Layer):
    def __init__(self):
        super(PositionalEmbeddingLayer,self).__init__()
        
    def build(self,input_shape):
        self.w = self.add_weight(shape=(1,)+input_shape[1:],initializer='glorot_uniform',trainable=True)
    
    def call(self,inputs):
        return inputs+self.w
    
    def get_config(self):
        config = {} 
        return config
    
    @classmethod
    def from_config(cls,config):
        return cls(**config)

###############################################################################    

class PatchingLayer(keras.layers.Layer):
    def __init__(self,patch_size):
        super(PatchingLayer,self).__init__()
        self.patch_size = patch_size
    
    def build(self,input_shape):
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        if height%self.patch_size != 0 or width%self.patch_size != 0:
            raise ValueError('PatchingLayer: patch_size is not compatible with image size')
            
        self.num_patches = height*width//self.patch_size**2
        self.patch_dim = channels*self.patch_size**2

            
    def call(self,inputs): #[batch_size,height,width,channels]
        batch_size = tf.shape(inputs)[0]

        patches = tf.image.extract_patches(images=inputs,sizes=[1, self.patch_size, self.patch_size, 1],
                                           strides=[1, self.patch_size, self.patch_size, 1],rates=[1, 1, 1, 1],
                                           padding="VALID") #[batch_size,row index,column index,patch_size*patch_size*channels]
        
        patches = tf.reshape(patches,[batch_size,self.num_patches,self.patch_dim])
        return patches #[batch_size,num_patches,patch_dim]
    
    def get_config(self):
        config = {'patch_size':self.patch_size}
        return config
    
    @classmethod
    def from_config(cls,config):
        return cls(**config)

###############################################################################

class ClassEmbeddingLayer(keras.layers.Layer):
    def __init__(self,embed_dim):
        super(ClassEmbeddingLayer,self).__init__()
        self.embed_dim = embed_dim
        
    def build(self,input_shape):
        self.w = self.add_weight(shape=[1,1,self.embed_dim],initializer='glorot_uniform',trainable=True)
    
    def call(self,inputs): #[batch_size,num_patches,embed_dim]
        batch_size = tf.shape(inputs)[0]
        
        class_emb = tf.broadcast_to(self.w,[batch_size,1,self.embed_dim]) #[batch_size,1,embed_dim]
        
        z = tf.concat([class_emb,inputs],1) #[batch_size,num_patches+1,embed_dim]
        
        return z #[batch_size,num_patches+1,embed_dim]
    
    def get_config(self):
        config = {'embed_dim':self.embed_dim}
        return config
    
    @classmethod
    def from_config(cls,config):
        return cls(**config)

###############################################################################    

class MultiHeadSelfAttentionLayer(keras.layers.Layer):
    def __init__(self,n_heads,embed_dim,weight_decay):
        super(MultiHeadSelfAttentionLayer,self).__init__()
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.weight_decay = weight_decay
        
        if self.embed_dim % self.n_heads != 0:
            raise ValueError('MultiHeadSelfAttentionLayer: embed_dim is not divisible by n_heads')
            
        self.proj_dim = self.embed_dim // self.n_heads
        
        self.d_q = Dense(self.embed_dim,activation=None,kernel_regularizer=L2(self.weight_decay))
        self.d_k = Dense(self.embed_dim,activation=None,kernel_regularizer=L2(self.weight_decay))
        self.d_v = Dense(self.embed_dim,activation=None,kernel_regularizer=L2(self.weight_decay))
        self.d_combined_heads = Dense(self.embed_dim,activation=None,kernel_regularizer=L2(self.weight_decay))
        
    def _attention(self,q,k,v): #[batch_size,n_heads,num_patches+1,proj_dim]
        score = tf.matmul(q,k,transpose_b=True) #[batch_size,n_heads,num_patches+1,num_patches+1]
        k_dim = tf.cast(tf.shape(k)[-1],tf.float32)
        score_n = score/tf.sqrt(k_dim) #[batch_size,n_heads,num_patches+1,num_patches+1]
        a_weights = tf.nn.softmax(score_n,axis=-1) #[batch_size,n_heads,num_patches+1,num_patches+1]
        a = tf.matmul(a_weights,v) #[batch_size,n_heads,num_patches+1,proj_dim]
        
        return a,a_weights
        
        
    def _split_heads(self,inputs): #[batch_size,num_patches+1,embed_dim]
        batch_size = tf.shape(inputs)[0]
        z = tf.reshape(inputs,[batch_size,-1,self.n_heads,self.proj_dim]) #[batch_size,num_patches+1,n_heads,proj_dim]
        z = tf.transpose(z,perm=[0,2,1,3]) #[batch_size,n_heads,num_patches+1,proj_dim]
        
        return z
    
    def call(self,inputs): #[batch_size,num_patches+1,embed_dim]
        batch_size = tf.shape(inputs)[0]
        
        q = self.d_q(inputs) #[batch_size,num_patches+1,embed_dim]
        k = self.d_k(inputs) #[batch_size,num_patches+1,embed_dim]
        v = self.d_v(inputs) #[batch_size,num_patches+1,embed_dim]
        
        q = self._split_heads(q) #[batch_size,n_heads,num_patches+1,proj_dim]
        k = self._split_heads(k) #[batch_size,n_heads,num_patches+1,proj_dim]
        v = self._split_heads(v) #[batch_size,n_heads,num_patches+1,proj_dim]
        
        a, a_weights = self._attention(q=q,v=v,k=k) #[batch_size,n_heads,num_patches+1,proj_dim], [batch_size,n_heads,num_patches+1,num_patches+1]
        
        a = tf.transpose(a,perm=[0,2,1,3]) #[batch_size,num_patches+1,n_heads,proj_dim]
        a_concat = tf.reshape(a,[batch_size,-1,self.embed_dim]) #[batch_size,num_patches+1,embed_dim]
        
        z = self.d_combined_heads(a_concat) #[batch_size,num_patches+1,embed_dim]
        
        return z
    
    def get_config(self):
        config = {'n_heads':self.n_heads,
                  'embed_dim':self.embed_dim,
                  'weight_decay':self.weight_decay}
        return config
    
    @classmethod
    def from_config(cls,config):
        return cls(**config)

###############################################################################

class TransformerBlock(keras.layers.Layer):
    def __init__(self,n_heads,embed_dim,dense_dim,dropout,weight_decay,ln_epsilon=1e-6):
        super(TransformerBlock,self).__init__()
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.ln_epsilon = ln_epsilon
        
        self.layer_norm_1 = LayerNormalization(epsilon=self.ln_epsilon)
        self.layer_norm_2 = LayerNormalization(epsilon=self.ln_epsilon)
        
        self.attention = MultiHeadSelfAttentionLayer(n_heads=self.n_heads,embed_dim=self.embed_dim,weight_decay=self.weight_decay)
        
        self.drop_1 = Dropout(self.dropout)
        self.drop_2 = Dropout(self.dropout)
        self.drop_3 = Dropout(self.dropout)
        
        self.d_embed = Dense(self.embed_dim,activation=None,kernel_regularizer=L2(self.weight_decay))
        self.d_dense = Dense(self.dense_dim,activation=tf.keras.activations.gelu,kernel_regularizer=L2(self.weight_decay))
            
    def call(self,inputs,training=None): #[batch_size,num_patches+1,embed_dim]
        z_1 = self.layer_norm_1(inputs) #[batch_size,num_patches+1,embed_dim]
        z_1 = self.attention(z_1) #[batch_size,num_patches+1,embed_dim]
        z_1 = self.drop_1(z_1,training=training) #[batch_size,num_patches+1,embed_dim] 
        res_1 = z_1 + inputs #[batch_size,num_patches+1,embed_dim]
        
        z_2 = self.layer_norm_2(res_1) #[batch_size,num_patches+1,embed_dim]
        
        z_2 = self.d_dense(z_2) #[batch_size,num_patches+1,dense_dim]
        z_2 = self.drop_2(z_2,training=training) #[batch_size,num_patches+1,dense_dim]
        z_2 = self.d_embed(z_2) #[batch_size,num_patches+1,embed_dim]
        z_2 = self.drop_3(z_2,training=training) #[batch_size,num_patches+1,embed_dim]
        res_2 = z_2 + res_1 #[batch_size,num_patches+1,embed_dim]
        return res_2
    
    def get_config(self):
        config = {'n_heads':self.n_heads,
                  'embed_dim':self.embed_dim,
                  'dense_dim':self.dense_dim,
                  'dropout':self.dropout,
                  'weight_decay':self.weight_decay,
                  'ln_epsilon':self.ln_epsilon}
        return config
    
    @classmethod
    def from_config(cls,config):
        return cls(**config)

###############################################################################

class ViTClassificationHead(keras.layers.Layer):
    def __init__(self,out_dim,dense_dim,dropout,weight_decay,ln_epsilon=1e-6):
        super(ViTClassificationHead,self).__init__()
        self.out_dim = out_dim
        self.dense_dim = dense_dim
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.ln_epsilon = ln_epsilon
        
        self.ln = LayerNormalization()
        self.d_1 = Dense(self.dense_dim,activation=tf.keras.activations.gelu,kernel_regularizer=L2(self.weight_decay))
        self.d_2 = Dense(self.out_dim,kernel_regularizer=L2(self.weight_decay))
        self.drop = Dropout(self.dropout)
        
    def call(self,inputs,training=None): #[batch_size,embed_dim]        
        #z = self.ln(inputs) #[batch_size,embed_dim]
        z = self.d_1(inputs) #[batch_size,dense_dim]
        z = self.drop(z,training=training) #[batch_size,dense_dim]
        z = self.d_2(z) #[batch_size,n_classes]
        
        return z
    
    def get_config(self):
        config = {'dense_dim':self.dense_dim,
                  'dropout':self.dropout,
                  'out_dim':self.out_dim,
                  'weight_decay':self.weight_decay,
                  'ln_epsilon':self.ln_epsilon}
        return config
    
    @classmethod
    def from_config(cls,config):
        return cls(**config)

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
    
class ViTMAE_encoder(keras.Model):
    def __init__(self,n_layers,n_heads,embed_dim,dense_dim,dropout,weight_decay,**kwargs):
        super(ViTMAE_encoder,self).__init__(**kwargs)
        
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.embed_dim = embed_dim 
        self.dense_dim = dense_dim
        self.dropout = dropout
        self.weight_decay = weight_decay
        
        self.transformer_blocks = dict([('Transformer_Block_'+str(i),
                                         TransformerBlock(n_heads=self.n_heads,
                                                          embed_dim=self.embed_dim,
                                                          dense_dim=self.dense_dim,
                                                          dropout=self.dropout,
                                                          weight_decay=self.weight_decay)) for i in range(self.n_layers)])
        
    def call(self,inputs,training=None): #[batch_size,num_patches,embedding_dim]
        z = inputs
        for key,layer in self.transformer_blocks.items():
            z = layer(z,training=training) #[batch_size,num_patches,embedding_dim]
                        
        return z #[batch_size,num_patches,embedding_dim]
            
    def get_config(self):
        config = {'n_layers':self.n_layers,
                  'n_heads':self.n_heads,
                  'embed_dim':self.embed_dim,
                  'dense_dim':self.dense_dim,
                  'dropout':self.dropout,
                  'weight_decay':self.weight_decay}
        return config
    
    @classmethod
    def from_config(cls,config):
        return cls(**config)

###############################################################################
###############################################################################

class ViTMAE_decoder(keras.Model):
    def __init__(self,n_layers,n_heads,embed_dim,dense_dim,patch_size,height,width,channels,dropout,weight_decay,ln_epsilon=1e-6,**kwargs):
        super(ViTMAE_decoder,self).__init__(**kwargs)
        
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.embed_dim = embed_dim 
        self.dense_dim = dense_dim
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.ln_epsilon = ln_epsilon
        self.patch_size = patch_size
        self.height = height
        self.width = width
        self.channels = channels
        self.image_dim = self.height*self.width*self.channels
        
        self.transformer_blocks = dict([('Transformer_Block_'+str(i),
                                         TransformerBlock(n_heads=self.n_heads,
                                                          embed_dim=self.embed_dim,
                                                          dense_dim=self.dense_dim,
                                                          dropout=self.dropout,
                                                          weight_decay=self.weight_decay)) for i in range(self.n_layers)])
        self.d1 = Dense(self.embed_dim,activation=None,kernel_regularizer=L2(self.weight_decay))
        self.ln = LayerNormalization(epsilon=self.ln_epsilon)
        self.d2 = Dense(self.patch_size**2*self.channels,activation=keras.activations.sigmoid,kernel_regularizer=L2(self.weight_decay))
        self.reshape = Reshape((self.height,self.width,self.channels))
        
    def call(self,inputs,training=None): #[batch_size,num_patches,embedding_dim_encoder]
        z = inputs
        
        z = self.d1(z) #[batch_size,num_patches,embedding_dim_decoder]
        
        for key,layer in self.transformer_blocks.items():
            z = layer(z,training=training) #[batch_size,num_patches,embedding_dim_decoder]
        
        z = self.ln(z) #[batch_size,num_patches,embedding_dim_decoder]
        
        z = self.d2(z) #[batch_size,num_patches,patch_dim]
        
        yhat = self.reshape(z)
                
        return yhat
            
    def get_config(self):
        config = {'n_layers':self.n_layers,
                  'n_heads':self.n_heads,
                  'embed_dim':self.embed_dim,
                  'dense_dim':self.dense_dim,
                  'patch_size':self.patch_size,
                  'height':self.height,
                  'width':self.width,
                  'channels':self.channels,
                  'dropout':self.dropout,
                  'weight_decay':self.weight_decay,
                  'ln_epsilon':self.ln_epsilon}
        return config
    
    @classmethod
    def from_config(cls,config):
        return cls(**config)

###############################################################################
    
class ViTMAE_PatchMaskEmbedEncodeLayer(keras.layers.Layer):
    def __init__(self,patch_size,embedding_dim,mask_p,weight_decay=0.0):
        super(ViTMAE_PatchMaskEmbedEncodeLayer,self).__init__()
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.weight_decay = weight_decay
        self.mask_p = mask_p
        
        self.patch = PatchingLayer(self.patch_size)
        
    def build(self,input_shape): #[batch_size,height,width,channels]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        if height%self.patch_size != 0 or width%self.patch_size != 0:
            raise ValueError('ViTMAE_PatchMaskEmbedEncodeLayer: patch_size is not compatible with image size')
        
        self.num_patches = height*width//self.patch_size**2
        self.patch_dim = channels*self.patch_size**2

        self.mask_token = self.add_weight(shape=[1, self.patch_dim],initializer='random_normal',trainable=True)
        self.linear_proj = Dense(self.embedding_dim,activation=None,kernel_regularizer=L2(self.weight_decay))
        self.positional_encoding = self.add_weight(shape=[1,self.num_patches,self.embedding_dim],initializer='glorot_uniform',trainable=True)
        
    def call(self,inputs,training=None): #[batch_size,height,width,channels]
        batch_size = tf.shape(inputs)[0]
        
        self.num_mask = tf.cast(self.mask_p*tf.cast(self.num_patches,tf.float32),tf.int32)         
        
        patches = self.patch(inputs) #[batch_size,num_patches,patch_dim]
        
        z = self.linear_proj(patches) #[batch_size,num_patches,embedding_dim]
        z = z + self.positional_encoding #[batch_size,num_patches,embedding_dim]
        
        if training:
            mask_indices,unmask_indices = self.get_random_indices(batch_size) #[batch_size,num_mask], [batch,num_unmask]
            
            unmasked_embeddings = tf.gather(z,unmask_indices,axis=1,batch_dims=1)  #[batch_size,num_unmask,embedding_dim]
            unmasked_positions = tf.gather(tf.tile(self.positional_encoding,[batch_size, 1, 1])
                                           ,unmask_indices,axis=1,batch_dims=1) #[batch_size,num_mask,embedding_dim]
            masked_positions = tf.gather(tf.tile(self.positional_encoding,[batch_size, 1, 1]),
                                         mask_indices, axis=1, batch_dims=1)  #[batch_size,num_mask,embedding_dim]
            
            mask_tokens = tf.repeat(self.mask_token,repeats=self.num_mask,axis=0) #[num_mask,patch_dim]
            mask_tokens = tf.repeat(mask_tokens[tf.newaxis, ...],repeats=batch_size,axis=0) #[batch_size,num_mask,patch_dim]
            
            masked_embeddings = self.linear_proj(mask_tokens) + masked_positions #[batch_size,num_mask,embedding_dim]
            
            return unmasked_embeddings,masked_embeddings,unmasked_positions,mask_indices,unmask_indices
            #unmasked_embeddings: [n,n_unmasked,embedding_dim]
            #^embedded and positionally encoded unmasked embeddings (encoder input)
            #masked_embeddings: [n,n_masked,embedding_dim]
            #^the embedded and positionally encoded mask tokens (half of decoder input)
            #unmasked_positions: [n,n_unmasked,embedding_dim]
            #^position embeddings of the unmasked_embeddings (added to encoder output before passing to decoder with masked_embeddings)
            #mask_indices: [n,n_masked]
            #^the indices that are masked for each observation (used for loss calc)
            #unmask_indices: [n,n_unmasked]
            #^the indices that are not masked for each observation (unused)
        else:
            return z #[batch_size,num_patches,embedding_dim]
    
    def get_random_indices(self, batch_size):
        rand_indices = tf.argsort(tf.random.uniform(shape=(batch_size, self.num_patches)), axis=-1) #[batch_size,num_patches]
        mask_indices = rand_indices[:, : self.num_mask] #[batch_size,num_mask]
        unmask_indices = rand_indices[:, self.num_mask :] #[batch,num_patches-num_mask]
        
        return mask_indices, unmask_indices #[batch_size,num_mask], [batch,num_patches-num_mask]
    
    def get_config(self):
        config = {'patch_size':self.patch_size,
                  'embedding_dim':self.embedding_dim,
                  'mask_p':self.mask_p,
                  'weight_decay':self.weight_decay}
        return config
    
    @classmethod
    def from_config(cls,config):
        return cls(**config)

###############################################################################

#output_size: size of the last layer of the projection head (ie the size of the vector that is operated on by the self supervised loss)
#proj_dim: proj dim - size of the first layer of the projection head
#embedding_dim: The ViT embedding size
#embed_dim: size of the last layer of the embedding model (ie the size of the input to the projection head)
#dense_dim: The ViT dense size

@keras_2.saving.register_keras_serializable()
class ViTMAE(keras.Model):
    def __init__(self,patch_size,n_layers_encoder,n_layers_decoder,n_heads_encoder,n_heads_decoder,embed_dim,embed_dim_decoder,dense_dim_encoder,dense_dim_decoder,mask_p,dropout,weight_decay,n_classes,n_regr,**kwargs):
        super(ViTMAE,self).__init__(**kwargs)
    
        self.patch_size = patch_size
        self.n_layers_encoder = n_layers_encoder
        self.n_layers_decoder = n_layers_decoder
        self.n_heads_encoder = n_heads_encoder
        self.n_heads_decoder = n_heads_decoder
        self.embed_dim = embed_dim
        self.embed_dim_decoder = embed_dim_decoder
        self.dense_dim_encoder = dense_dim_encoder
        self.dense_dim_decoder = dense_dim_decoder
        self.mask_p = mask_p
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.n_classes = n_classes
        self.n_regr = n_regr
        
        self.patch = PatchingLayer(self.patch_size)
        
        self.patch_embed_encode_mask = ViTMAE_PatchMaskEmbedEncodeLayer(patch_size=self.patch_size,
                                                 embedding_dim=self.embed_dim,
                                                 mask_p=self.mask_p,
                                                 weight_decay=0.0)
        
        self.encoder = ViTMAE_encoder(n_layers=self.n_layers_encoder,
                                      n_heads=self.n_heads_encoder,
                                      embed_dim=self.embed_dim,
                                      dense_dim=self.dense_dim_encoder,
                                      dropout=self.dropout,
                                      weight_decay=self.weight_decay)
                
        self.classifier_probes = []
        for n_class in self.n_classes:
            self.classifier_probes.append(Dense(n_class))
        
        self.regressor_probes = []
        for i in range(self.n_regr):
            self.regressor_probes.append(Dense(1))
            
        self.gap = GlobalAveragePooling1D()
    
    def build(self,input_shape):
        self.decoder = ViTMAE_decoder(n_layers=self.n_layers_decoder,
                                      n_heads=self.n_heads_decoder,
                                      embed_dim=self.embed_dim_decoder,
                                      dense_dim=self.dense_dim_decoder,
                                      patch_size=self.patch_size,
                                      height=input_shape[1],
                                      width=input_shape[2],
                                      channels=input_shape[3],
                                      dropout=self.dropout,
                                      weight_decay=self.weight_decay)
    
    def call(self,inputs,training=None): #[batch_size,height,width,channels]
        if training:
            unmasked_embeddings,masked_embeddings,unmasked_positions,mask_indices,unmask_indices = self.patch_embed_encode_mask(inputs,training=training)
            
            unmasked_encoder_outputs = self.encoder(unmasked_embeddings,training=training) #[batch_size,num_unmasked,embedding_dim]
            
            ###################################################################
            unmasked_decoder_inputs = unmasked_encoder_outputs + unmasked_positions #[batch_size,num_unmasked,embedding_dim]
            decoder_inputs = tf.concat([unmasked_decoder_inputs, masked_embeddings], axis=1) #[batch_size,num_patches,embedding_dim]
            
            decoder_outputs = self.decoder(decoder_inputs) #[batch_size,height,width,channels]
            patched_decoder_outputs = self.patch(decoder_outputs) #[batch_size,num_patches,patch_dim]
            
            patches = self.patch(inputs) #[batch_size,num_patches,patch_dim]
            y = tf.gather(patches, mask_indices, axis=1, batch_dims=1) #[batch_size,num_mask,patch_dim]
            yhat = tf.gather(patched_decoder_outputs, mask_indices, axis=1, batch_dims=1) #[batch_size,num_mask,patch_dim]
            ###################################################################
            
            classifier_preds = []
            for classifier in self.classifier_probes:
                classifier_preds.append(classifier(tf.stop_gradient(self.gap(unmasked_encoder_outputs))))
            regressor_preds = []
            for regressor in self.regressor_probes:
                regressor_preds.append(regressor(tf.stop_gradient(self.gap(unmasked_encoder_outputs))))
            
            return y,yhat,classifier_preds,regressor_preds 
            #[batch_size, num_mask, patch_dim]
            #[batch_size, num_mask, patch_dim]
            #list([n,n_classes[i]])
            #list([n,1])
        else:
            unmasked_embeddings = self.patch_embed_encode_mask(inputs,training=training) #[batch_size,num_patches,embedding_dim]
            unmasked_encoder_outputs = self.encoder(unmasked_embeddings,training=training) #[batch_size,num_patches,embedding_dim]
            
            classifier_preds = []
            for classifier in self.classifier_probes:
                classifier_preds.append(classifier(tf.stop_gradient(self.gap(unmasked_encoder_outputs))))
            regressor_preds = []
            for regressor in self.regressor_probes:
                regressor_preds.append(regressor(tf.stop_gradient(self.gap(unmasked_encoder_outputs))))
            
            return unmasked_encoder_outputs,classifier_preds,regressor_preds 
            #[batch_size,num_patches,embedding_dim]
            #list([n,n_classes[i]])
            #list([n,1])
        
    def get_config(self): 
        config = {'patch_size':self.patch_size,
                  'n_layers_encoder':self.n_layers_encoder,
                  'n_layers_decoder':self.n_layers_decoder,
                  'n_heads_encoder':self.n_heads_encoder,
                  'n_heads_decoder':self.n_heads_decoder,
                  'embed_dim':self.embed_dim,
                  'embed_dim_decoder':self.embed_dim_decoder,
                  'dense_dim_encoder':self.dense_dim_encoder,
                  'dense_dim_decoder':self.dense_dim_decoder,
                  'mask_p':self.mask_p,
                  'dropout':self.dropout,
                  'weight_decay':self.weight_decay,
                  'n_classes':self.n_classes,
                  'n_regr':self.n_regr}
        return config
    
    @classmethod
    def from_config(cls,config):
        return cls(**config)
    
###############################################################################

#out_dim: The dimension of the last Dense layer of the classification_head
#proj_dim: The dimension of the embedding layer used downstream
#n_class: The dimension of the linear_probe Dense layer
class ViT(keras.Model):
    def __init__(self,patch_size,n_layers,n_heads,embed_dim,dense_dim,out_dim,proj_dim,dropout,weight_decay,n_class,**kwargs):
        super(ViT,self).__init__(**kwargs)
        
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.out_dim = out_dim
        self.proj_dim = proj_dim
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.n_class = n_class
        
        self.patch = PatchingLayer(patch_size=self.patch_size)
        self.linear_proj = Dense(self.embed_dim,activation=None,kernel_regularizer=L2(self.weight_decay))
        self.class_emb = ClassEmbeddingLayer(self.embed_dim)
        self.pos_emb = PositionalEmbeddingLayer()
        
        self.transformer_blocks = dict([('Transformer_Block_'+str(i),
                                         TransformerBlock(n_heads=self.n_heads,
                                                          embed_dim=self.embed_dim,
                                                          dense_dim=self.dense_dim,
                                                          dropout=self.dropout,
                                                          weight_decay=self.weight_decay)) for i in range(self.n_layers)])
        
        self.embed_resize = Dense(self.proj_dim,activation=tf.keras.activations.gelu)
        
        self.classification_head = ViTClassificationHead(out_dim=self.out_dim,
                                                         dense_dim=self.proj_dim,
                                                         dropout=self.dropout,
                                                         weight_decay=self.weight_decay)
        
        self.linear_probe = Dense(self.n_class)
        
        
    def call(self,inputs,training=None): #[batch_size,height,width,channels]
                
        patches = self.patch(inputs) #[batch_size,num_patches,patch_dim]
        
        z = self.linear_proj(patches) #[batch_size,num_patches,embed_dim]
        
        z = self.class_emb(z) #[batch_size,num_patches+1,embed_dim]
        
        z = self.pos_emb(z) #[batch_size,num_patches+1,embed_dim]
        
        for key,layer in self.transformer_blocks.items():
            z = layer(z,training=training) #[batch_size,num_patches+1,embed_dim]
        
        z_N0N = self.embed_resize(z[:,0,:])
        
        logits = self.classification_head(z_N0N,training=training) #[batch_size,n_classes]
        
        logits_probe = self.linear_probe(tf.stop_gradient(z))
        
        return z_N0N,logits,logits_probe
        
    def get_config(self):
        config = {'patch_size':self.patch_size,
                  'n_layers':self.n_layers,
                  'n_heads':self.n_heads,
                  'embed_dim':self.embed_dim,
                  'dense_dim':self.dense_dim,
                  'out_dim':self.out_dim,
                  'proj_dim':self.proj_dim,
                  'dropout':self.dropout,
                  'weight_decay':self.weight_decay,
                  'n_class':self.n_class}
        return config
    
    @classmethod
    def from_config(cls,config):
        return cls(**config)
