# CS 331 - Spring 2020
# Programming Assignment 3 - Sentiment Analysis
# Junhyeok Jeong, jeongju@oregonstate.edu
# Youngjoo Lee, leey3@oregonstate.edu

import math
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import re
import pandas as pd

def clean_text(text):

    # remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # remove the characters [\], ['] and ["]
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\"", "", text)
    #pattern = r'[^a-zA-z0-9\s]'
    #text = re.sub(pattern, '', text)
    text = re.sub(r'\d+', "", text)

    # convert text to lowercase
    text = text.strip().lower()

    # replace punctuation characters with spaces
    filters = '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = dict((c, " ") for c in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)

    return text

def create_bow(input, vocabulary):
    vectorizer = CountVectorizer(
        analyzer = "word",
        tokenizer=None,
        preprocessor=clean_text,
        stop_words=None,
        vocabulary=vocabulary,
        max_features=2000
    )
    bag_of_words = vectorizer.fit_transform(input)
    bag_of_words = bag_of_words.toarray()
    
    name = vectorizer.get_feature_names()

    return bag_of_words, name

if __name__ == "__main__":
    # Importing the dataset
    with open("trainingSet.txt", "r") as f:
        sentences = []
        labels = []
    #read and copy every line on an array called 'lines'
        for i, line in enumerate(f):
            split = line.strip().split('\t')
            sentences.append(clean_text(split[0]))
            labels.append(int(split[1]))

    print(sentences)
    print(labels)
    
    vectorizer = CountVectorizer(
    stop_words="english"
    )

    # fit the vectorizer on the text
    vectorizer.fit(sentences)
    # get the vocabulary
    inv_vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
    vocabulary = [inv_vocab[i] for i in range(len(inv_vocab))]
    print(vocabulary)

    # 1. generate BOW for all restaurant reviews
    print("generating BOW for restaurant reviews ...")
    bow, name = create_bow(sentences, vocabulary)
    bow = np.sum(bow, axis=0)

    model = pd.DataFrame( 
    (count, word) for word, count in
    zip(bow, name))
    model.columns = ['Word', 'Count']
    print(model)
    print("done ... !")
    