# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 18:24:48 2021

@author: Maximus
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


df = pd.read_csv(r'DataSet/words.csv')

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
  
warnings.filterwarnings(action = 'ignore')

import gensim
from gensim.models import Word2Vec


print(df['0'])
data = []

for i in df['0']:
    for j in word_tokenize(i):
        data.append(i)
    
# Create CBOW model
model2 = gensim.models.Word2Vec(data, min_count = 1, 
                              vector_size = 100, window = 5 , sg =1)


print("words: " , df['0'][0] , "    and    "  , df['0'][1])

# Print results
print("Cosine similarity between 'alice' " +
          "and 'wonderland' - Skip Gram : ",
    model2)
     

print("words: " , df['0'][0] , "    and    "  , df['0'][1000])
print("Cosine similarity between 'alice' " +
            "and 'machines' - Skip Gram : ",
    model2.similarity(df['0'][0] , df['0'][1000]))
