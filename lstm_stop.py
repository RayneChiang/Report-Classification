from tkinter import *

import numpy
import pandas as pd
import numpy as np
import os
import re
import nltk

nltk.download('punkt')
nltk.download('wordnet')
import tkinter as tk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Input, LSTM, Dense, Bidirectional, Dropout, Activation
from keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from tkinter.ttk import *
import timeit


def testFilter(df):
    for index, row in df.iterrows():
        des = row['Description  ']
        if (des.find('TEST') != -1 or des.find('test') != -1 or des.find('Test') != -1 or des.isnumeric()) and row[
            'Flag'] == 'No':
            df = df.drop(index)
    df = df.drop_duplicates()
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


def text_to_sequence(df, size, len):
    # Tokenize the mails
    tok = Tokenizer(num_words=size)
    nna_report = extract_word(df)  # 1442 left with words
    nna_report.reset_index(drop=True, inplace=True)
    y = nna_report['Flag'].map({'Yes': 1, 'No': 0}).values
    tok.fit_on_texts(nna_report['Words'])
    # Use text_to_sequence to convert it into vectors
    sequences = tok.texts_to_sequences(nna_report['Words'])

    # pad seqence to create a matrix of equal length mails
    X = sequence.pad_sequences(sequences, maxlen=len)
    return X, y, nna_report['Words']


def train():
    start = timeit.default_timer()
    dir_name = os.path.dirname(__file__)
    all_report = pd.read_excel(dir_name + '/input/Stop App Reports no password.xlsx', sheet_name='All Reports - 4022')
    nna_report = all_report.dropna().copy()  # 2978 records left, use copy leaves the origin data unchanged
    nna_report['Description  '] = nna_report['Description  '].astype(str)
    nna_report = testFilter(nna_report)  # 1459 left

    vocab_size = 10000
    max_len = 250

    X, y, df = text_to_sequence(nna_report[:1200], vocab_size, max_len)
    df.to_csv(dir_name + '/test/training_data.csv')

    vocab_size = 10000
    max_len = 250
    checkpoint_filepath = dir_name + '/content/tmp/checkpoint'
    model = Sequential()
    model.add(Embedding(vocab_size, 200, input_length=max_len))
    model.add(LSTM(32))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='max',
        save_best_only=True)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0)
    model.fit(X, y, validation_split=0.2, shuffle=True, epochs=30, batch_size=64,
              callbacks=[earlyStopping, model_checkpoint_callback])

    model.save(dir_name + "/model/my_model")
    stop = timeit.default_timer()
    Used_time = str(round(stop - start))
    label.configure(text="Training Finished in " + Used_time + " seconds")
    print('Time: ', stop - start)


def generate_test_dataset():
    T.delete(1.0, END)
    dir_name = os.path.dirname(__file__)
    all_report = pd.read_excel(dir_name + '/input/Stop App Reports no password.xlsx', sheet_name='All Reports - 4022')
    nna_report = all_report.dropna().copy()  # 2978 records left, use copy leaves the origin data unchanged
    select_records = nna_report[1200:].sample(1)
    select_records = select_records['Description  '].values[0]
    T.insert(tk.END, select_records)
    str_test.set(select_records)
    label.configure(text="Ready for test")


def generate_wordlist(str):
    lemmatizer = WordNetLemmatizer()
    str = str.lower()
    tokens = word_tokenize(str)
    words = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha()]  # Lemmatization
    return words


def test():
    dir_name = os.path.dirname(__file__)
    str = str_test.get()
    print(str)
    if str.isnumeric():
        label.configure(text="Human not look")
        return
    else:
        load_model = keras.models.load_model(dir_name + "/model/my_model")
        word_list = generate_wordlist(str)
        print(word_list)
        if len(word_list) == 0:
            label.configure(text="Human not look")
            return
        df = pd.read_csv(dir_name + '/test/training_data.csv')
        tok = Tokenizer(num_words=10000)
        df = df.append({'Words': word_list}, ignore_index=True)
        tok.fit_on_texts(df['Words'])
        sequences = tok.texts_to_sequences(df['Words'])
        test_data = sequence.pad_sequences(sequences, maxlen=250)
        print(len(test_data))
        probs = load_model.predict_proba(test_data)
        thresh = 0.1
        pred = [1 if prob > thresh else 0 for prob in probs]
        print(probs[-1])
        if pred[-1] == 0:
            label.configure(text="Human not look")
        else:
            label.configure(text="Human should look")




if __name__ == '__main__':
    root = tk.Tk()
    root.title("Stop App")
    root.geometry('500x300')
    frame = tk.Frame(root)
    frame.pack()

    label = tk.Label(root, text="Ready", bg='white', fg='black', font=('Arial', 12), width=30, height=2)
    label.pack(padx=20, pady=20, side=BOTTOM)

    btn_train = tk.Button(frame,
                          text="Train",
                          command=train)
    btn_train.pack(padx=5, pady=10, side=LEFT)

    str_test = tk.StringVar()
    btn_generate_test = tk.Button(frame,
                                  text="Generate Test Dataset",
                                  command=generate_test_dataset)
    btn_generate_test.pack(padx=5, pady=10, side=LEFT)

    btn_train = tk.Button(frame,
                          text="Test",
                          command=test)
    btn_train.pack(padx=5, pady=10, side=LEFT)

    T = tk.Text(root, height=10, width=50)
    T.pack(padx=10, pady=10)

    root.mainloop()




    # fpr, tpr, threshold = metrics.roc_curve(y_test, y_prob)
    # roc_auc = metrics.auc(fpr, tpr)
    #
    # plt.title('Receiver Operating Characteristic')
    # plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    # plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.savefig('Roc curve.jpg')

    # thresh = 0.1
    # pred = [1 if prob > thresh else 0 for prob in y_prob]
    # print(metrics.accuracy_score(y_test, pred))
    # print('Confusion Matrix', confusion_matrix(y_test, pred))
    # model.save("sst")
