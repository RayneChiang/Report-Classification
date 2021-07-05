from time import time

import numpy as np
import pandas as pd
import os
import nltk

nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
import pickle
import tensorflow_decision_forests as tfdf
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from gensim.models.phrases import Phrases, Phraser
import multiprocessing
from gensim import models
from matplotlib import pyplot as plt
from wordcloud import WordCloud
def testFilter(df):
    for index, row in df.iterrows():
        des = row['Description  ']
        if des.find('TEST') != -1 or des.find('test') != -1 or des.find('Test') != -1 or des.isnumeric():
            df = df.drop(index)
    return df


def extract_word(df):
    lemmatizer = WordNetLemmatizer()
    word_list = []
    for index, row in df.iterrows():
        des = row['Description  ']
        des = des.lower()
        tokens = word_tokenize(des)
        words = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha()]  # Lemmatization
        if len(words) == 0:
            df = df.drop(index)
        else:
            word_list.append(words)
    df['Words'] = word_list
    return df


# def generate_dictionary(dataframe):
#     vectorizer = TfidfVectorizer()
#     X = vectorizer.fit_transform(dataframe['Description  '])
#     print(vectorizer.get_feature_names())
#     print(vectorizer.get_stop_words())

def train_dt(dataframe):
    nna_report = extract_word(dataframe)
    y = nna_report['Flag'].map({'Yes': 1, 'No': 0}).values
    cv = TfidfVectorizer()
    X = cv.fit_transform(nna_report['Description  ']).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    filename = 'decision_tree.sav'
    pickle.dump(clf, open(filename, 'wb'))
    dt_pred = clf.predict(X_test)
    print(accuracy_score(dt_pred, y_test))
    print('confusion matrix:\n', confusion_matrix(y_test, dt_pred))
    return dt_pred, y_test


# word to vec functions
def word2idx(word):
        return w2v_model.wv.vocab[word].index

def idx2word(idx):
        return w2v_model.wv.index2word[idx]

# def train_dt(dataframe):
#     nna_report = extract_word(dataframe)
#     y = nna_report['Flag'].map({'Yes': 1, 'No': 0}).values
#     cv = CountVectorizer()
#     X = cv.fit_transform(nna_report['Description  ']).toarray()
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
#     clf = tree.DecisionTreeClassifier()
#     clf.fit(X_train, y_train)
#     filename = 'decision_tree.sav'
#     pickle.dump(clf, open(filename, 'wb'))
#     dt_pred = clf.predict(X_test)
#     print(accuracy_score(dt_pred, y_test))
#     return dt_pred

def train_lstm(dataframe):
    # Tokenize behaves worse than Bag of word
    vocab_size = 10000
    max_len = 250
    # Tokenize the mails
    tok = Tokenizer(num_words=vocab_size)
    nna_report = extract_word(dataframe)
    y = nna_report['Flag'].map({'Yes': 1, 'No': 0}).values
    tok.fit_on_texts(nna_report['Words'])
    # Use text_to_sequence to convert it into vectors
    sequences = tok.texts_to_sequences(nna_report['Words'])
    # pad seqence to create a matrix of equal length mails
    sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len)
    X_train, X_test, y_train, y_test = train_test_split(sequences_matrix, y, test_size=0.2, random_state=1)
    vocab_size = 10000
    max_len = 250
    lstm_model = Sequential()
    lstm_model.add(Embedding(vocab_size, 200, input_length=max_len))
    lstm_model.add(LSTM(32))
    lstm_model.add(Dense(1, activation='sigmoid'))
    lstm_model.summary()
    lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    lstm_model.fit(X_train, y_train, epochs=10, batch_size=64)
    lstm_model.save('lstm_model')
    scores = lstm_model.evaluate(X_test, y_test, verbose=0)
    print(scores[0])
    print(scores[1])
    y_pred_lstm = lstm_model.predict_classes(X_test)
    print('confusion matrix:\n', confusion_matrix(y_test, y_pred_lstm))
    return y_pred_lstm, y_test


if __name__ == '__main__':
    dir_name = os.path.dirname(__file__)
    all_report = pd.read_excel(dir_name + '/input/Stop App Reports no password.xlsx', sheet_name='All Reports - 4022')
    nna_report = all_report.dropna().copy()  # 2978 records left, use copy leaves the origin data unchanged
    nna_report['Description  '] = nna_report['Description  '].astype(str)
    nna_report = testFilter(nna_report)  # 1459 left
    nna_report = extract_word(nna_report)
    # human_look = nna_report[nna_report['Flag'] == 1]
    # human_ignore = nna_report[nna_report['Flag'] == 0]

    # wordCloyud
    # text = ''
    # for news in human_ignore['Description  '].values:
    #     text += f" {news}"
    # wordcloud = WordCloud(
    #     width=3000,
    #     height=2000,
    #     background_color='black',
    #     # stopwords=set(nltk.corpus.stopwords.words("english"))).generate(str(text))
    # )
    # fig = plt.figure(
    #     figsize=(40, 30),
    #     facecolor='k',
    #     edgecolor='k')
    # plt.imshow(wordcloud, interpolation='bilinear')
    # plt.axis('off')
    # plt.tight_layout(pad=0)
    # plt.show()
    # del text

    phrases = Phrases(nna_report['Words'], min_count=30, progress_per=10000)
    bigram = Phraser(phrases)
    sentences = bigram[nna_report['Words']]
    # # most frequent word
    # word_freq = defaultdict(int)
    # for sent in sentences:
    #     for i in sent:
    #         word_freq[i] += 1
    # len(word_freq)
    # print(sorted(word_freq, key=word_freq.get, reverse=True)[:10])

    cores = multiprocessing.cpu_count()
    print(cores)
    # w = models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    w2v_model = models.Word2Vec(min_count=20,
                         window=2,
                         sample=6e-5,
                         alpha=0.03,
                         min_alpha=0.0007,
                         negative=20,
                         workers=cores - 1)
    w2v_model.build_vocab(sentences, progress_per=10000)
    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
    pretrained_weights = w2v_model.prepare_weights()
    vocab_size, emdedding_size = pretrained_weights.shape


    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size,
                        weights=[pretrained_weights]))
    model.add(LSTM(units=emdedding_size))
    model.add(Dense(units=vocab_size))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy')


    # Tokenize the mails
    tok = Tokenizer(num_words=vocab_size)
    y = nna_report['Flag'].map({'Yes': 1, 'No': 0}).values
    tok.fit_on_texts(nna_report['Words'])
    # Use text_to_sequence to convert it into vectors
    sequences = tok.texts_to_sequences(nna_report['Words'])
    # pad seqence to create a matrix of equal length mails
    sequences_matrix = sequence.pad_sequences(sequences, maxlen=250)
    X_train, X_test, y_train, y_test = train_test_split(sequences_matrix, y, test_size=0.2, random_state=1)
    model.fit(X_train, y_train, epochs=10, batch_size=64)
    scores = model.evaluate(X_test, y_test, verbose=0)
    print(scores[0])
    print(scores[1])

    # pred_lstm, y_lstm = train_lstm(nna_report)
    # pred_dt, y_dt = train_dt(nna_report)
    # pred = pred_lstm.T + pred_dt.T
    # pred[pred > 1] = 1
    # print('confusion matrix:\n', confusion_matrix(y_lstm, pred.T))
    #
    # print(np.count_nonzero(pred_lstm.T-pred_dt.T))
    # print(np.count_nonzero(y_lstm.T-y_dt.T))
    # print('Test loss:', scores[0])
    # print('Test accuracy:', scores[1])
