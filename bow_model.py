import pandas as pd
import os
import re
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


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
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer()
    X = cv.fit_transform(nna_report['Description  ']).toarray()
    y = nna_report['Flag'].map({'Yes': 1, 'No' : 0}).values

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True)

    # Fitting Naive Bayes classifier to the Training set
    from sklearn.naive_bayes import MultinomialNB

    classifier = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    '''
    Confusion Matrix
    array([[863,  11],
           [  1, 264]])
    '''
    # this function computes subset accuracy
    from sklearn.metrics import accuracy_score

    print(accuracy_score(y_test, y_pred))  # 0.9894644424934153
    print(accuracy_score(y_test, y_pred, normalize=False))  # 1129 out of 1139

    # Applying k-Fold Cross Validation
    from sklearn.model_selection import cross_val_score

    accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train,
                                 cv=10)
    print(accuracies.mean())
    print(accuracies.std())
