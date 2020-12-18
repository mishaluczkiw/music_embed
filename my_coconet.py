# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 23:27:49 2020

@author: Misha Luczkiw and Nick Triano
"""

import tensorflow as tf
import pickle
import numpy as np

# load embedding
weights = pickle.load(open('weights_filtered.pckl','rb'))

# load vocab hashes
vocab_hash = pickle.load(open('vocab_hash_filtered.pckl','rb'))
vocab_hash_list = list(vocab_hash) # easier for indexing

# load vocab slices
vocab_filtered = pickle.load(open('vocab_filtered.pckl','rb'))

# load word_list
word_list = pickle.load(open('word_list_filtered.pckl','rb'))


# divide word list into equal chunks of music
piece_len = 50
append_len = len(word_list)%piece_len
wl_arr = np.concatenate((np.array(word_list),np.zeros(append_len)))



wl_arr = np.reshape(wl_arr,(-1,50))
# load inverse_vocab
# inverse_vocab3 = pickle.load(open('inverse_vocab3_1000.pckl','rb'))

num_pieces = wl_arr.shape[0]
embedding_size = weights.shape[1]
#%% Create first embedded piece
n_pieces = 300
piece1 = np.empty((n_pieces,piece_len,embedding_size))

for piece in range(n_pieces):
    for row in range(piece_len):
        # piece1[piece,row,:] = weights[vocab_hash_list.index(word_list[row]),:]
        piece1[piece,row,:] = weights[vocab_hash_list.index(wl_arr[piece,row]),:]

        

#%% Create architecture



xin = tf.keras.Input(shape=(piece_len,embedding_size,1))
# myLayer = MaskLayer((xin.shape))

# x = myLayer(xin)


num_conv_layers = 4
conv_filts = [32,64,128,256,256]
for i in range(num_conv_layers):
    if i == 0:
        # x = tf.keras.layers.BatchNormalization()(xin)
        x = tf.keras.layers.Dropout(rate = 0.5, noise_shape=(None,None,None,1),
                                    trainable=True)(xin)*0.5
        # x = drop_layer2(xin,training=True)*0.5
    # else: continue
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters = conv_filts[i],kernel_size=3,
                               activation='tanh',padding='same',
                               kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))(x)

xout = tf.keras.layers.Conv2D(filters=1,kernel_size=1,activation='tanh',padding='same')(x)
# xout = tf.keras.layers.Softmax()(x)

print(xout.shape)
model = tf.keras.Model(inputs=xin,outputs=xout)
print(model.summary())

#%%

loss_fn = tf.keras.losses.MeanSquaredError()
# loss_fn = tf.keras.losses.CategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam(lr=1e-3)

# model.compile(loss=loss_fn,optimizer=optimizer,metrics = ['accuracy'])
model.compile(loss=loss_fn,optimizer=optimizer,metrics = [tf.keras.metrics.MeanSquaredError()])


# model.fit(tf.expand_dims(masked_piece,0),tf.expand_dims(piece1,0),batch_size=1,epochs=10)
# model.fit(piece1,piece1,batch_size=10,epochs=10)

model.fit(piece_masked,piece1,batch_size=1,epochs=100)



# model.fit(tf.expand_dims(piece1,0),tf.expand_dims(piece1,0),batch_size=1,epochs=10)
#%% Test model
# piece_out= model.evaluate(tf.expad_dims(piece1[0,:,:],0),tf.expand_dims(piece1[0,:,:],0))

piece_out = model.predict(piece1)
piece_out = np.squeeze(piece_out)
print(piece_out.shape)
print(np.sum(piece_out))
print(np.sum(piece1))
print(loss_fn(piece1,piece_out).numpy())

#%% Generate random song
from sklearn.metrics.pairwise import cosine_similarity

p_in = np.random.normal(size=(1,piece_len,embedding_size,1),scale=0.3,loc =0)

for iterations in range(20):
    
    p_in = model.predict(p_in)
    
#%% Convert embeddings to slices by looking at nearest neighbor
slice_rand = []

for iRandSlice in range(piece_len):
    emb = p_in[0,iRandSlice,:,0]
    sim = cosine_similarity(emb.reshape(1, -1), weights).flatten()
    i_closest = int(np.where(sim == np.amax(sim[sim != np.amax(sim)]))[0])
    slice_rand.append(vocab_filtered[i_closest])
    




