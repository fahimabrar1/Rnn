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
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
  
warnings.filterwarnings(action = 'ignore')

import gensim
from gensim.models import Word2Vec






df = pd.read_excel(r'DataSet/words.xlsx')

lst = ['car', 'dog']


"""
        
    The problem lies in list, run the code and check the data list.
    items are not assigned/appended as string.
    thus the embedding can't be processed futher.

    Check the screen shot or list in repository.
    Hope you find the difference and the solution.
    
    Thank me later :) 

"""


data = []

for i in df[0]:
    for j in word_tokenize(i):
        data.append(j)
        
        
model = gensim.models.Word2Vec(data, min_count = 1, 
                              vector_size = 100, window = 5 , sg =1)

model.train(data, total_examples=model.corpus_count, epochs=model.epochs)

print("print: ",model.wv.similarity(data[0] , data[1]))

      


