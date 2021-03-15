from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, GRU
from keras.layers import Embedding
from keras.layers import Input
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
from keras.datasets import imdb
import sys,os
from numpy import array
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt

from nltk.corpus import stopwords                   #Stopwords corpus
from nltk.stem import PorterStemmer 
import re, string
from nltk.tokenize import word_tokenize 
stemmer = PorterStemmer() 
import pandas as pd
import gensim 
from gensim.models import Word2Vec 
from tqdm import tqdm
from nltk.tokenize import WordPunctTokenizer
from collections import Counter
from keras.layers.core import Reshape, Flatten
from os import listdir
from numpy import array
from numpy import asarray
from numpy import zeros
import keras
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, accuracy_score, classification_report
from wordcloud import WordCloud
import seaborn as sn

import matplotlib.pyplot as plt
import numpy as np
import itertools


def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True):
      

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    plt.clf()


df = pd.read_csv("data/processed/ds.csv", sep=',')

# print(len(df))
# sys.exit(0)
# 50000, 100000, 200000, 300000, 400000

len_data = 400000

#90/10
df = df.head(int(len_data * 0.90))
df_test = df.tail(int(len_data * 0.10))

# print(len(df))
# sys.exit(0)

# # x_train = df["tweet"]
# y_category = df[['health/medicine','business/finance','education/school','travel/tourism','science/technology']].values
y_category = df[['health/medicine','business/finance','education/school']].values

y_sentiment = df["sentiment"]
sentiment_labels = to_categorical(y_sentiment)

# print(y_sentiment)
# print(sentiment_labels)
# sys.exit(0)



train_df = df



'''
# objects = ('health', 'business', 'education', 'travel', 'technology')
objects = ('health', 'business', 'education')
y_pos = np.arange(len(objects))
# performance = [train_df['health/medicine'].sum(),train_df['business/finance'].sum(),train_df['education/school'].sum(),train_df['travel/tourism'].sum(),train_df['science/technology'].sum()]
performance = [train_df['health/medicine'].sum(),train_df['business/finance'].sum(),train_df['education/school'].sum()]
# plt.bar(y_pos, performance, align='center',  color=('r','g','b','m','y'))
plt.bar(y_pos, performance, align='center',  color=('r','g','b'))
plt.xticks(y_pos, objects)
plt.xticks(rotation=25)
plt.ylim(0, 400000)
y_ticks = np.arange(0, 400000, 100000)
plt.yticks(y_ticks)
plt.ylabel('Number')
plt.title('Event Distribution')
plt.show()


sentiment_group = df.groupby(["sentiment"])["sentiment"].count()

# print(sentiment_group)
# sys.exit(0)
# # objects = ('Positive', 'Negative', 'Neutral')
objects = ('Vey Negative', 'Negative', 'Neutral','Positive', 'Very Positive')

y_pos = np.arange(len(objects))
# performance = [sentiment_group[1],sentiment_group[0],sentiment_group[2]]
performance = [sentiment_group[1],sentiment_group[0],sentiment_group[2],sentiment_group[3],sentiment_group[4]]
# plt.bar(y_pos, performance, align='center', color=('r','g','b'))
plt.bar(y_pos, performance, align='center', color=('r','g','b','m','y'))
plt.xticks(y_pos, objects)
plt.xticks(rotation=25)
plt.ylim(0, 400000)
y_ticks = np.arange(0, 400000, 100000)
plt.yticks(y_ticks)
plt.ylabel('Number')
plt.title('Sentiment Distribution')
plt.show()


sys.exit(0)
'''



punkTokenizer = WordPunctTokenizer()
vocab = Counter()

stopWords = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'} 
def process_comments(list_sentences):
    comments = []
    for text in tqdm(list_sentences):
        split_txt = punkTokenizer.tokenize(text)
        filtered_sentence = [w for w in split_txt if not w in stopWords] 
        # print(split_txt)
        vocab.update(filtered_sentence)
        comments.append(filtered_sentence)
    return comments

total_tweet = process_comments(train_df['text'])
print("The vocabulary contains {} unique tokens".format(len(vocab)))


# # WordClod implementation
# wordcloud_text = ' ' .join(train_df['tweet'].values)
# wordcloud = WordCloud(width = 800, height = 800, 
#             background_color ='white', 
#             min_font_size = 10).generate(wordcloud_text) 

# # plot the WordCloud image                        
# plt.figure(figsize = (8, 8), facecolor = None) 
# plt.imshow(wordcloud) 
# plt.axis("off") 
# plt.tight_layout(pad = 0) 
# plt.show() 

# MAX_NB_WORDS = len(word_vectors.vocab)
MAX_NB_WORDS = 400000
# FEATURES = 300#len(word_vectors.vocab)
EMBEDDING_DIM=300
MAX_SEQUENCE_LENGTH = max([len(i) for i in total_tweet])
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(total_tweet)


X = tokenizer.texts_to_sequences(total_tweet)
# sequences_valid=tokenizer.texts_to_sequences(tweet_list_val)


word_index = tokenizer.word_index
# define vocabulary size (largest integer value)
vocab_size = len(tokenizer.word_index) + 1

# X = sequence.pad_sequences(X)
X = sequence.pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH, padding="pre", truncating="post")

# train = df
# x_train = train["tweet"]
# y_train_category = train_df[['health/medicine','business/finance','education/school','travel/tourism','science/technology']].values
y_train_category = train_df[['health/medicine','business/finance','education/school']].values

y_train_sentiment = train_df["sentiment"]
sentiment_labels = to_categorical(y_train_sentiment)



# x_train, x_test, y_train, y_test = train_test_split(X, y_train_category, test_size=0.1, random_state=42)
# _, _, z_train, z_test = train_test_split(X, sentiment_labels, test_size=0.1, random_state=42)


embeddings_index = {}
f = open('embedding/glove.6B.300d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

numMultiLabel = len(y_train_category[0])

BATCH_SIZE = 32
embedding_dims = 300
filters = 32
kernel_size = 24
hidden_dims = 128
EPOCH = 20

sentiment_labels = to_categorical(y_train_sentiment)

# sys.exit(0)
print('Loading data...')

print(" ================================================= ")

print('Build model...')



input_main = Input(shape=(X.shape[1],))

# create the embedding layer
x = Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)(input_main)
 

# x = (Dropout(0.2))(x)

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
# x = Conv1D(filters,kernel_size,padding='valid',activation='relu',strides=1)(x)
# we use max pooling:
# x = GlobalMaxPooling1D()(x)


# use this
x = Conv1D(32,kernel_size=3,padding='same',activation='relu')(x)
x = MaxPooling1D(pool_size=3)(x)
x = Dropout(0.2)(x)


# x = Conv1D(64,kernel_size=3,padding='same',activation='relu')(x)
# x = MaxPooling1D(pool_size=3)(x)
# x = Dropout(0.35)(x)
# x = Conv1D(128,kernel_size=3,padding='same',activation='relu')(x)
# x = MaxPooling1D(pool_size=3)(x)
# x = Dropout(0.4)(x)

x = LSTM(50,return_sequences=True)(x)
x = Dropout(0.2)(x)


x = Flatten()(x)


# We add a vanilla hidden layer:
x = Dense(128)(x)
# x = Dropout(0.2) (x)

# x = Dense(64)(x)
# x = Dropout(0.2) (x)


x = Activation('relu')(x)


output1 = Dense(numMultiLabel, activation = 'sigmoid', name='output1')(x)
output2 = Dense(5, activation = 'softmax', name='output2')(x)

multi_model = Model(inputs=input_main,outputs=[output1,output2])


opt = keras.optimizers.Adam(lr=0.01)

multi_model.compile(optimizer='adam',
              loss={'output1': 'binary_crossentropy', 'output2': 'categorical_crossentropy'},
              loss_weights={'output1': 0.6, 'output2': 0.7}, metrics=['accuracy'])

es = EarlyStopping(monitor='loss', verbose=1, patience=3)

# And trained it via:
history = multi_model.fit(X,{'output1': y_train_category, 'output2': sentiment_labels}, epochs=EPOCH, batch_size=BATCH_SIZE, callbacks=[es])
# history = multi_model.fit(x_train,{'output1': y_train, 'output2': z_train}, epochs=EPOCH, batch_size=BATCH_SIZE, validation_data=(x_test,{'output1': y_test, 'output2': z_test}), callbacks=my_callbacks)

plot_model(multi_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
# print(history.history)

# Plot training & validation accuracy values
plt.plot(history.history['output1_accuracy'])
plt.plot(history.history['output2_accuracy'])
plt.title('Training Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.ylim(0, 1)
plt.xlim(0, EPOCH)
plt.legend(['category', 'polarity'], loc='upper left')
plt.show()
# plt.clf()
print("===================================Accuracy Output 1=================================================")
print(history.history['output1_accuracy'])
print("====================================================================================")

print("===================================Accuracy Output 2=================================================")
print(history.history['output2_accuracy'])
print("====================================================================================")

# Plot training & validation loss values
plt.plot(history.history['output1_loss'])
plt.plot(history.history['output2_loss'])
plt.title('Model Training loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.xlim(0, EPOCH)
plt.ylim(0, 1)
plt.legend(['category', 'polarity'], loc='upper left')
plt.show()
# plt.clf()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.title('Training loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.ylim(0, 1)
plt.xlim(0, EPOCH)
plt.legend(['loss', 'val_loss'], loc='upper left')
plt.show()
# plt.clf()
print("===================================Loss=================================================")
print(history.history['loss'])
print("====================================================================================")



# y_cat_pred = df_test[['health/medicine','business/finance','education/school','travel/tourism','science/technology']].values
# y_list = df_test[['health/medicine','business/finance','education/school','travel/tourism','science/technology']].columns

y_cat_pred = df_test[['health/medicine','business/finance','education/school']].values
y_list = df_test[['health/medicine','business/finance','education/school']].columns

y_cat_sentiment = df_test["sentiment"]
sentiment_labels = to_categorical(y_cat_sentiment)
tweet = df_test['text']

seq = tokenizer.texts_to_sequences(tweet)
padded = sequence.pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)

print("===============================================================================================================")
print("Event Category Result")
print("===============================================================================================================")


(sigmoid_res, softmax_res) = multi_model.predict(padded)

sigmoid_res = np.round(sigmoid_res)



res = multilabel_confusion_matrix(y_cat_pred, sigmoid_res)
print(">>>><<<<<>>>><<<>>>>><<<<>>><<<<>>>><<<<<>>>><<<<>>>><<<<>>>")

for i,r in enumerate(res):
    plot_confusion_matrix(cm   =  r, 
                      normalize    = False,
                      target_names = ['Yes', 'No'],
                      title        = y_list[i] + " Confusion Matrix")


# cr = classification_report(y_cat_pred, sigmoid_res, target_names=['health', 'business', 'education','travel','science'])
cr = classification_report(y_cat_pred, sigmoid_res, target_names=['health', 'business', 'education'])

print(cr)

acc = accuracy_score(y_cat_pred, sigmoid_res)
print(acc)



print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("Polarity Result")
print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

print(softmax_res)

softmax_res = np.argmax(softmax_res, axis=1)
# print(softmax_res)
# print(y_train_sentiment.values)
acc2 = accuracy_score(y_cat_sentiment.values, softmax_res)
print(acc2)

cr2 = classification_report(y_cat_sentiment.values, softmax_res)
print(cr2)

res2 = confusion_matrix(y_cat_sentiment.values, softmax_res)
print(res2)

plot_confusion_matrix(cm   =  res2, 
                      normalize    = False,
                      target_names = ['Vey Negative', 'Negative', 'Neutral','Positive', 'Very Positive'],
                      title        = "Polarity Confusion Matrix")