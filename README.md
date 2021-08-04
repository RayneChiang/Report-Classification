# Triage demo
The project to create a ranking system for reports received from the stop App. Here is the demo shows how it works. 

Training process:
![training_gif](https://i.postimg.cc/R0JnjV77/train.gif)

Testing process:
![test.gif](https://i.postimg.cc/B6DH1QtS/test.gif)

## Model Detail

### Dataset
There are 1459 records of data available for training in total, one out of four of them should be classified as "Human should Look" and I split the dataset and 80% were used for training and 20% were used for testing.

### Preprocessing
* Normalization:  lower case the words
* Stemming and Lemmatization: get the base word and remove all the prefix and suffix 
* Tokenization: split the sentences in a more clever way

### LSTM
The input of the LSTM model is a sequence of words and the output is the predicted probability so that we can set a threshold to reduce the False-Negative cases.  LSTM model performs well on solving text classification problems since it can understand context based on the recent dependency. 
![lstm_explained](https://miro.medium.com/max/2000/1*0hkR4Bqiq1MN6Mew8E9t1w.png)

### Performance
The AUC score is 0.98 evaluated on the test dataset which is pretty good.

<img src="https://i.postimg.cc/wMgyyXS2/performance.png" style="zoom:150%;" />






