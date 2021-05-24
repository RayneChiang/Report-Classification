import pandas as pd
import os
import re
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
# Fitting Naive Bayes classifier to the Training set
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
import xgboost
from sklearn import svm, tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score,train_test_split

from sklearn.feature_extraction.text import CountVectorizer

import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding,Input,LSTM,Dense,Bidirectional,Dropout, Activation
from keras.models import Model
from tensorflow.keras.models import Sequential

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


if __name__ == '__main__':
    dir_name = os.path.dirname(__file__)
    all_report = pd.read_excel(dir_name + '/input/Stop App Reports no password.xls', sheet_name='All Reports - 4022')
    nna_report = all_report.dropna().copy()  # 2978 records left, use copy leaves the origin data unchanged
    nna_report['Description  '] = nna_report['Description  '].astype(str)
    nna_report = testFilter(nna_report)  # 1459 left
    # nna_report = extract_word(nna_report)  # 1432 left with words

    # Try applying simple bag of model directly

    cv = CountVectorizer()
    X = cv.fit_transform(nna_report['Description  ']).toarray()
    y = nna_report['Flag'].map({'Yes': 1, 'No' : 0}).values

    # Splitting the dataset into the Training set and Test set


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True)




    # # Making the Confusion Matrix
    # from sklearn.metrics import confusion_matrix
    # cm = confusion_matrix(y_test, y_pred)
    # print(cm)
    # '''
    # Confusion Matrix
    # array([[863,  11],
    #        [  1, 264]])
    # '''
    # this function computes subset accuracy

    models = [RandomForestClassifier(),
              GaussianNB(),
              AdaBoostClassifier(),
              xgboost.XGBClassifier(),
              svm.SVC(),
              tree.DecisionTreeClassifier(),
              KNeighborsClassifier(),
              MultinomialNB()]

    model_names = ['Random Forest Classifier',
                   'Gaussian Naive Bayes Classifier',
                   'Adaboost Classifier',
                   'XGBoost Classifier',
                   'Support Vector Classifier',
                   'Decision Tree Classifier',
                   'K Nearest Neighbour Classifier',
                   'Multinomial Naive Bayes Classifier']
    accuracy_mean = []
    accuracy_std = []
    d = {}
    for model in range(len(models)):
        clf = models[model]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracies = cross_val_score(estimator=clf, X=X_train, y=y_train,
                                     cv=10)
        mean = accuracies.mean()
        std = accuracies.std()
        accuracy_mean.append(mean)
        accuracy_std.append(std)
    d = {'Modelling Name': model_names, 'Accuracy_mean': accuracy_mean, 'Accuracy_std': accuracy_std}
    accuracy_frame = pd.DataFrame.from_dict(d, orient='index').transpose()
    print(accuracy_frame)








