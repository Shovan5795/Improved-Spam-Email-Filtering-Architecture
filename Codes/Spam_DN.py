# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 10:00:03 2021

@author: shovon5795
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 20:15:31 2021

@author: shovon5795
"""

import pandas as pd
import re
import datetime



data = pd.read_csv(r"cleaned_spam_email.csv")

data['Label'].value_counts()

#Plot function
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Vs Validation Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Vs Validation loss')
    plt.legend()




X=data.drop('Label',axis=1)
y = data['Label']

messages = X.copy()
messages['Spam Email']= messages['Spam Email'].apply(str)
corpus = []

for i in range (0, len(messages)):
    review = messages['Spam Email'][i]
    corpus.append(review)




from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from nltk.tokenize import word_tokenize
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.layers import MaxPool1D



all_words = []



for sentence in corpus:
    tokenize_word = word_tokenize(sentence)
    for word in tokenize_word:
        all_words.append(word)

unique = set(all_words)

voc_size = len(unique)

one_hot_representation = [one_hot(sent, voc_size) for sent in  corpus]

word_count = lambda sentence: len(word_tokenize(sentence))
longest_sentence = max(corpus, key=word_count)
length_long_sentence = len(word_tokenize(longest_sentence))

embedded_doc = pad_sequences(one_hot_representation, padding = 'pre', maxlen = length_long_sentence)

embedding_vector_features = 40
model = Sequential()
model.add(Embedding(voc_size, embedding_vector_features, input_length = length_long_sentence))
start_time = datetime.datetime.now()

'''
#Artificial Neural Network

model.add(Dense(16, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation= 'relu'))
model.add(Dropout(0.4))
model.add(Dense(8, activation = 'relu'))
model.add(Flatten())

'''


#LSTM#BLSTM#GRU#SimpleRNN
#model.add(GRU(8))
model.add(LSTM(8))
#model.add(SimpleRNN(8))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())


'''

#CNN
model.add(Conv1D(filters = 40, kernel_size = 5, strides = 1, activation = 'relu', padding = 'same'))
model.add(Conv1D(filters = 20, kernel_size = 3, strides = 1, activation = 'relu', padding = 'same'))
model.add(Conv1D(filters = 10, kernel_size = 3, strides = 1, activation = 'relu', padding = 'same'))
model.add(GlobalMaxPooling1D())



#CNN-LSTM
model.add(Conv1D(filters = 40, kernel_size = 5, strides = 1, activation = 'relu', padding = 'same'))
model.add(Conv1D(filters = 20, kernel_size = 3, strides = 1, activation = 'relu', padding = 'same'))
model.add(Dropout(0.2))
model.add(MaxPool1D())
model.add(Conv1D(filters = 10, kernel_size = 3, strides = 1, activation = 'relu', padding = 'same'))
model.add(Conv1D(filters = 8, kernel_size = 3, strides = 1, activation = 'relu', padding = 'same'))
model.add(MaxPool1D())
model.add(LSTM(8, return_sequences = True))
model.add(LSTM(8))
'''


import numpy as np

X_final=np.array(embedded_doc)
y_final=np.array(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)



history = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=512)

end_time = datetime.datetime.now()
time_diff = (end_time - start_time)
execution_time = time_diff.total_seconds() * 1000
print(execution_time/10)

plot_history(history)

y_pred = model.predict_classes(X_test)


from sklearn.metrics import confusion_matrix, accuracy_score,classification_report, roc_auc_score
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test, y_pred))