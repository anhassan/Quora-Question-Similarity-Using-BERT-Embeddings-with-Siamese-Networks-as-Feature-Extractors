# -*- coding: utf-8 -*-
import sys
import os 
import pandas as pd
import numpy as np
from tqdm import tqdm
from time import time

import numpy as np
from numpy import genfromtxt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import keras
from keras import backend as K
from keras import callbacks
import tensorflow as tf
import tensorflow_hub as hub


from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Activation, BatchNormalization, regularizers, Add, concatenate, Layer,Lambda
from keras.optimizers import RMSprop, SGD, Adam
from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import Conv1D , MaxPooling1D, Flatten,Dense,Input,Lambda
from keras.layers import LSTM, Concatenate, Dropout
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras import backend as K
from keras import regularizers

def add_features():
  data_features = pd.read_csv("data/features_from_BERT_Vectors.csv")
  features = data_features.drop(['question1', 'question2', 'is_duplicate','jaccard_distance'],axis=1).values
  return features

def create_base_network_cnn(input_dimensions):

    input  = Input(shape=(input_dimensions[0],input_dimensions[1]))
    conv1  = Conv1D(filters=32,kernel_size=8,strides=1,activation = 'relu',name='conv1')(input)
    b1 = BatchNormalization()(conv1)
    d1 = Dropout(0.1)(b1)
    
    pool1  = MaxPooling1D(pool_size=1,strides=1,name='pool1')(d1)
    d2  = Dropout(0.1)(pool1)
    
    conv2  = Conv1D(filters=64,kernel_size=6,strides=1,activation = 'relu',name='conv2')(d2)
    b2 = BatchNormalization()(conv2)
    d3 = Dropout(0.1)(b2)
    
    pool2  = MaxPooling1D(pool_size=1,strides=1,name='pool2')(d3)
    d4 = Dropout(0.1)(pool2)
    
    conv3  = Conv1D(filters=128,kernel_size=4,strides=1,activation = 'relu',name='conv3')(d4)
    b3 = BatchNormalization()(conv3)
    d4 = Dropout(0.1)(b3)
    
    
    pool3  = MaxPooling1D(pool_size=1,strides=1,name='pool3')(d4)
    d5 = Dropout(0.1)(pool3)
    
    flat   = Flatten(name='flat_cnn')(d5)
    d1 = Dense(100, activation='sigmoid',kernel_regularizer=regularizers.l2(0.01))(flat)
    drop1 = Dropout(0.1)(d1)
    b1 = BatchNormalization()(drop1)
    d2 = Dense(25,kernel_regularizer=regularizers.l2(0.01))(b1)
    drop2 = Dropout(0.1)(d2)
    b2 = BatchNormalization()(drop2)
    d2 = Dense(5,kernel_regularizer=regularizers.l2(0.01))(b2)
    drop3 = Dropout(0.1)(d2)
    bn = BatchNormalization()(drop3)

    model  = Model(input=input,output=bn)
  
  
    return model


def dense_network(features1):
    input = Input(shape=(features1[0],))
    #x = Flatten()(features)
    d1 = Dense(100, activation='sigmoid',kernel_regularizer=regularizers.l2(0.01))(input)
    drop1 = Dropout(0.1)(d1)
    b1 = BatchNormalization()(drop1)
    d2 = Dense(25,kernel_regularizer=regularizers.l2(0.01))(b1)
    drop2 = Dropout(0.1)(d2)
    b2 = BatchNormalization()(drop2)
    d2 = Dense(5,kernel_regularizer=regularizers.l2(0.01))(b2)
    drop3 = Dropout(0.1)(d2)
    b3 = BatchNormalization()(drop3)
#     flat   = Flatten(name='flat_dnn2')(b3)
    model = Model(input = input,output=b3)
    return model
  

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)



def create_network(input_dimensions,num_features):

#     # #Fasttext
    base_network_lstm_1 = dense_network([324,1])
    input_a_lstm_1 = Input(shape=(input_dimensions[0],))
    input_b_lstm_1 = Input(shape=(input_dimensions[0],))
    # LSTM with embedding 1
    inter_a_lstm_1 = base_network_lstm_1(input_a_lstm_1)
    inter_b_lstm_1 = base_network_lstm_1(input_b_lstm_1)
    d_lstm_1 = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([inter_a_lstm_1, inter_b_lstm_1])


    #W2V
    base_network_lstm_2 = dense_network([324,1])
    input_a_lstm_2 = Input(shape=(input_dimensions[0],))
    input_b_lstm_2 = Input(shape=(input_dimensions[0],))
    # LSTM with embedding 2
    inter_a_lstm_2 = base_network_lstm_2(input_a_lstm_2)
    inter_b_lstm_2 = base_network_lstm_2(input_b_lstm_2)
    d_lstm_2 = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([inter_a_lstm_2, inter_b_lstm_2])


    #Glove
    base_network_lstm_3 = dense_network([324,1])
    input_a_lstm_3 = Input(shape=(input_dimensions[0],))
    input_b_lstm_3 = Input(shape=(input_dimensions[0],))
    # LSTM with embedding 3
    inter_a_lstm_3 = base_network_lstm_3(input_a_lstm_3)
    inter_b_lstm_3 = base_network_lstm_3(input_b_lstm_3)
    d_lstm_3 = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([inter_a_lstm_3, inter_b_lstm_3])

    #Uncomment to Use BERT as Siamese as well
#     base_network_lstm_4 = dense_network([768,1])
#     input_a_lstm_4 = Input(shape=(768,))
#     input_b_lstm_4 = Input(shape=(768,))
#     # LSTM with embedding 3
#     inter_a_lstm_4 = base_network_lstm_4(input_a_lstm_4)
#     inter_b_lstm_4 = base_network_lstm_4(input_b_lstm_4)


#     CNN
    base_network_cnn = create_base_network_cnn(input_dimensions)
    # CNN with 3 channel embedding
    input_a_cnn = Input(shape=(input_dimensions[0],input_dimensions[1]))
    input_b_cnn = Input(shape=(input_dimensions[0],input_dimensions[1]))
    inter_a_cnn = base_network_cnn(input_a_cnn)
    inter_b_cnn = base_network_cnn(input_b_cnn)



    d_cnn = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([inter_a_cnn, inter_b_cnn])
    d_lstm_1 = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([inter_a_lstm_1, inter_b_lstm_1])
    d_lstm_2 = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([inter_a_lstm_2, inter_b_lstm_2])
    d_lstm_3 = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([inter_a_lstm_3, inter_b_lstm_3])
#     d_lstm_4 = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([inter_a_lstm_4, inter_b_lstm_4])

    # Additional Features from (BERT)
    features = Input(shape=(num_features,))

    #BERT itself
    features_b = Input(shape=(768,))


    #Concatenation of Features
    feature_set = Concatenate(axis=-1)([d_cnn,d_lstm_1,d_lstm_2,d_lstm_3,features,features_b])




    #x = Flatten()(features)
    d1 = Dense(300, activation='relu',kernel_regularizer=regularizers.l2(0.01))(feature_set)
    drop1 = Dropout(0.1)(d1)
    b1 = BatchNormalization()(drop1)
    d2 = Dense(20, activation='relu',kernel_regularizer=regularizers.l2(0.001))(b1)
    drop2 = Dropout(0.1)(d2)
    b2 = BatchNormalization()(drop2)
    d3 = Dense(2, activation='softmax',kernel_regularizer=regularizers.l2(0.01))(b2)


#     model = Model(input=[feature_set,features_b], output=d3)
    model = Model(input=[input_a_cnn, input_b_cnn , input_a_lstm_1, input_b_lstm_1, input_a_lstm_2, input_b_lstm_2, input_a_lstm_3, input_b_lstm_3,features,features_b], output=d3)  
    print("Model Architecture Designed")
    return model
  

def main():
    #Get Dataset
  df_sub = pd.read_csv('data/data.csv')
  print('Shape of Dataset',df_sub.shape)

  
  df_sub['question1'] = df_sub['question1'].apply(lambda x: str(x))
  df_sub['question2'] = df_sub['question2'].apply(lambda x: str(x))
  q1sents = list(df_sub['question1'])
  q2sents = list(df_sub['question2'])
  tokenized_q1sents = [word_tokenize(i) for i in list(df_sub['question1'])]
  tokenized_q2sents = [word_tokenize(i) for i in list(df_sub['question2'])]


  print('Loading Embeddings W2vec')
  w2v_emb_q1 = genfromtxt('data/Embeddings/word2vec/w2vec_q1_balanced.csv', delimiter=',',skip_header=1)
  w2v_emb_q2 = genfromtxt('data/Embeddings/word2vec/w2vec_q2_balanced.csv', delimiter=',',skip_header=1)
  w2v_emb_q1 = np.delete(w2v_emb_q1, 0, 1)
  w2v_emb_q2 = np.delete(w2v_emb_q2, 0, 1)
  print('Loading Embeddings fastext')
  ft_emb_q1 = genfromtxt('data/Embeddings/fastext/fastext_q1_balanced.csv', delimiter=',',skip_header=1)
  ft_emb_q2 = genfromtxt('data/Embeddings/fastext/fastext_q2_balanced.csv', delimiter=',',skip_header=1)
  ft_emb_q1 = np.delete(ft_emb_q1, 0, 1)
  ft_emb_q2 = np.delete(ft_emb_q2,0, 1)
  print('Loading Embeddings glove')
  glove_emb_q1 = genfromtxt('data/Embeddings/glove/glove_q1_balanced.csv', delimiter=',',skip_header=1)
  glove_emb_q2 = genfromtxt('data/Embeddings/glove/glove_q2_balanced.csv', delimiter=',',skip_header=1)
  glove_emb_q1 = np.delete(glove_emb_q1,0, 1)
  glove_emb_q2 = np.delete(glove_emb_q2, 0, 1)
  print('Loading Embeddings BERT')
  bert_q = genfromtxt('data/Embeddings/bert/bert_qpair_balanced.csv', delimiter=',',skip_header=1)
  bert_q = np.delete(bert_q,0,1)
  bert_q1 = genfromtxt('data/Embeddings/bert/bert_q1_balanced.csv', delimiter=',',skip_header=1)
  bert_q1 = np.delete(bert_q1,0,1)
  print('Loading Embeddings BERTQ2')
  bert_q2 = genfromtxt('data/Embeddings/bert/bert_q2_balanced.csv', delimiter=',',skip_header=1)
  bert_q2 = np.delete(bert_q2,0,1)

  features = add_features()

  num_train = int(df_sub.shape[0] * 0.70)
  num_val = int(df_sub.shape[0] * 0.10)
  num_test = df_sub.shape[0] - num_train - num_val 
              
  print("Number of training pairs: %i"%(num_train))
  print("Number of Validation pairs: %i"%(num_val))
  print("Number of testing pairs: %i"%(num_test))


    # init data data arrays
  X_train_cnn_a = np.zeros([num_train, 324, 3])
  X_test_cnn_a  = np.zeros([num_test, 324, 3])
  X_val_cnn_a  = np.zeros([num_val, 324, 3])

  X_train_cnn_b = np.zeros([num_train, 324, 3])
  X_test_cnn_b  = np.zeros([num_test, 324, 3])
  X_val_cnn_b  = np.zeros([num_val, 324, 3])

  Y_train = np.zeros([num_train]) 
  Y_test = np.zeros([num_test])
  Y_val = np.zeros([num_val]) 


  #Labels
  Y_train = df_sub['is_duplicate'].values[num_train]
  Y_val = df_sub['is_duplicate'].values[num_val]
  Y_test = df_sub['is_duplicate'].values[num_val]
  


  num_val = num_train + int(df_sub.shape[0] * 0.10)
  # fill data arrays with features
  X_train_cnn_a[:,:,0] = ft_emb_q1[:num_train]
  X_train_cnn_a[:,:,1] = w2v_emb_q1[:num_train]
  X_train_cnn_a[:,:,2] = glove_emb_q1[:num_train]

  X_train_cnn_b[:,:,0] = ft_emb_q2[:num_train]
  X_train_cnn_b[:,:,1] = w2v_emb_q2[:num_train]
  X_train_cnn_b[:,:,2] = glove_emb_q2[:num_train]

  features_train = features[:num_train]
  features_b_train = bert_q[:num_train]
  Y_train = df_sub[:num_train]['is_duplicate'].values

  X_val_cnn_a[:,:,0] = ft_emb_q1[num_train:num_val]
  X_val_cnn_a[:,:,1] = w2v_emb_q1[num_train:num_val]
  X_val_cnn_a[:,:,2] = glove_emb_q1[num_train:num_val]

  X_val_cnn_b[:,:,0] = ft_emb_q2[num_train:num_val]
  X_val_cnn_b[:,:,1] = w2v_emb_q2[num_train:num_val]
  X_val_cnn_b[:,:,2] = glove_emb_q2[num_train:num_val]

  features_val = features[num_train:num_val]
  features_b_val = bert_q[num_train:num_val]
  Y_val = df_sub[num_train:num_val]['is_duplicate'].values


  X_test_cnn_a[:,:,0] = ft_emb_q1[num_val:]
  X_test_cnn_a[:,:,1] = w2v_emb_q1[num_val:]
  X_test_cnn_a[:,:,2] = glove_emb_q1[num_val:]

  X_test_cnn_b[:,:,0] = ft_emb_q2[num_val:]
  X_test_cnn_b[:,:,1] = w2v_emb_q2[num_val:]
  X_test_cnn_b[:,:,2] = glove_emb_q2[num_val:]
  features_test = features[num_val:]
  features_b_test = bert_q[num_val:]
  Y_test = df_sub[num_val:]['is_duplicate'].values


  Y_train = keras.utils.to_categorical(Y_train, num_classes=2)
  Y_test = keras.utils.to_categorical(Y_test, num_classes=2)
  Y_val = keras.utils.to_categorical(Y_val, num_classes=2)

 
  
  
  #############Pipeline Starts################
  net = create_network([324,3],25)
  optimizer = Adam(lr=0.001)
  net.compile(loss="binary_crossentropy", optimizer=optimizer,metrics=['accuracy'])
  print(net.summary())


  # # Training Sets for LSTM1

  X_intera_train_1 = X_train_cnn_a[:,:,0]
  X_interb_train_1 = X_train_cnn_b[:,:,0]
  X_train_lstm1_a = X_intera_train_1
  X_train_lstm1_b = X_interb_train_1

  X_intera_val_1 = X_val_cnn_a[:,:,0]
  X_interb_val_1 = X_val_cnn_b[:,:,0]
  X_val_lstm1_a = X_intera_val_1
  X_val_lstm1_b = X_interb_val_1

  X_intera_test_1 = X_test_cnn_a[:,:,0]
  X_interb_test_1 = X_test_cnn_b[:,:,0]
  X_test_lstm1_a = X_intera_test_1
  X_test_lstm1_b = X_interb_test_1


  # Validation Sets for LSTM2

  X_intera_train_2 = X_train_cnn_a[:,:,1]
  X_interb_train_2 = X_train_cnn_b[:,:,1]
  X_train_lstm2_a = X_intera_train_2
  X_train_lstm2_b = X_interb_train_2

  X_intera_val_2 = X_val_cnn_a[:,:,1]
  X_interb_val_2 = X_val_cnn_b[:,:,1]
  X_val_lstm2_a = X_intera_val_2
  X_val_lstm2_b = X_interb_val_2

  X_intera_test_2 = X_test_cnn_a[:,:,1]
  X_interb_test_2 = X_test_cnn_b[:,:,1]
  X_test_lstm2_a = X_intera_test_2
  X_test_lstm2_b = X_interb_test_2
  
  # Test Set for LSTM3

  X_intera_train_3 = X_train_cnn_a[:,:,2]
  X_interb_train_3 = X_train_cnn_b[:,:,2]
  X_train_lstm3_a = X_intera_train_3
  X_train_lstm3_b = X_interb_train_3
 
  X_intera_val_3 = X_val_cnn_a[:,:,2]
  X_interb_val_3 = X_val_cnn_b[:,:,2]
  X_val_lstm3_a = X_intera_val_3
  X_val_lstm3_b = X_interb_val_3
  
  X_intera_test_3 = X_test_cnn_a[:,:,2]
  X_interb_test_3 = X_test_cnn_b[:,:,2]
  X_test_lstm3_a = X_intera_test_3
  X_test_lstm3_b = X_interb_test_3


  # Test Set for LSTM4 BERT

  # X_intera_train_4 = bert_q1[:num_train]
  # X_train_lstm4_a = X_intera_train_4

  # X_intera_val_4 = bert_q1[num_train:num_val]
  # X_val_lstm4_a = X_intera_val_4

  # X_intera_test_4 = bert_q1[num_val:]
  # X_test_lstm4_a = X_intera_test_4

  # X_interb_train_4 = bert_q2[:num_train]
  # X_train_lstm4_b = X_interb_train_4

  # X_interb_val_4 = bert_q2[num_train:num_val]
  # X_val_lstm4_b = X_interb_val_4

  # X_interb_test_4 = bert_q2[num_val:]
  # X_test_lstm4_b = X_interb_test_4

  print("Input Shapes")
  print("CNN Shape")
  print(X_train_cnn_a.shape,X_val_cnn_a.shape,X_test_cnn_a.shape)
  print("LSTM (x3) Shape:")
  print(X_train_lstm1_a.shape,X_val_lstm1_a.shape,X_test_lstm1_a.shape)

  print("Features shape:",features_train.shape,features_val.shape,features_test.shape)
  print("BERT Features shape:",features_b_train.shape,features_b_val.shape,features_b_test.shape)
  
  print("Labels Shape")
  print(Y_train.shape,Y_val.shape,Y_test.shape)

  filepath="./QQP_{epoch:02d}_{val_loss:.4f}.h5"
  checkpoint = callbacks.ModelCheckpoint(filepath, 
                                        monitor='val_loss', 
                                        verbose=0, 
                                        save_best_only=True)
  callbacks_list = [checkpoint]
  

  net.fit([X_train_cnn_a, X_train_cnn_b, X_train_lstm1_a, X_train_lstm1_b,X_train_lstm2_a, X_train_lstm2_b,
          X_train_lstm3_a, X_train_lstm3_b,features_train,features_b_train], 
          Y_train,
          validation_data=([X_val_cnn_a, X_val_cnn_b,X_val_lstm1_a, X_val_lstm1_b,
          X_val_lstm2_a, X_val_lstm2_b,X_val_lstm3_a, X_val_lstm3_b,features_val,features_b_val]
                          , Y_val),
          batch_size=384, nb_epoch=100, shuffle=True,callbacks = callbacks_list)
  score = net.evaluate([X_test_cnn_a, X_test_cnn_b,X_test_lstm1_a, X_test_lstm1_b,X_test_lstm2_a, X_test_lstm2_b,X_test_lstm3_a, X_test_lstm3_b,features_test,features_b_test],Y_test,batch_size=384)
  # score = net.evaluate([X_test_cnn_a, X_test_cnn_b,X_test_lstm1_a, X_test_lstm1_b,
  #               X_test_lstm2_a, X_test_lstm2_b,X_test_lstm3_a, X_test_lstm3_b,X_test_lstm4_a, X_test_lstm4_b,features_test,features_b_test],Y_test,batch_size=384)
  print('Test loss : {:.4f}'.format(score[0]))
  print('Test accuracy : {:.4f}'.format(score[1]))
  return 0


if __name__== "__main__":
  main()





