#You need a BERT server running to use this script
#In order to get BERT Server Running please refer to this link below:
# https://github.com/hanxiao/bert-as-service


import sys
import os
import numpy as np_array
import pandas as pd
from bert_serving.client import BertClient

def get_bertembeddings_pair(q1,q2):
  q = list(map(lambda x, y: x+ ' ||| ' +y, q1, q2))
  bc = BertClient(check_length=False)
  bert_embeddings = bc.encode(q)
  return bert_embeddings

def get_bertembeddings_sent(q):
  bc = BertClient(check_length=False)
  bert_embeddings = bc.encode(q)
  return bert_embeddings

def main():
  #Code for Balanced Data
  df_sub = pd.read_csv('data/data.csv')
  
  print('Shape of Dataset',df_sub.shape)

  df_sub['question1'] = df_sub['question1'].apply(lambda x: str(x))
  df_sub['question2'] = df_sub['question2'].apply(lambda x: str(x))
  q1sents_b = list(df_sub['question1'])
  q2sents_b = list(df_sub['question2'])

  print("Getting Bert Embeddings pairs")
  bert_b = get_bertembeddings_pair(q1sents_b,q2sents_b)
  pd.DataFrame(bert_b).to_csv("bert_qpair_balanced.csv")

  print("Getting Bert Embeddings Q1.")
  bert_b_q1 = get_bertembeddings_sent(q1sents_b)
  pd.DataFrame(bert_b_q1).to_csv("bert_q1_balanced.csv")

  print("Getting Bert Embeddings Q2-balanced")
  bert_b_q2 = get_bertembeddings_sent(q2sents_b)
  pd.DataFrame(bert_b_q2).to_csv("bert_q2_balanced.csv")

if __name__== "__main__":
  main()
