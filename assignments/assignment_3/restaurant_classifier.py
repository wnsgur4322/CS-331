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
        analyzer = 'word'
    )
    bag_of_words = vectorizer.fit_transform(input)
    bag_of_words = bag_of_words.toarray()
    
    name = vectorizer.get_feature_names()

    return bag_of_words, name


######### fix this to Bernoulli version ##############
# To calculate the total number of words in reviews (positive or negative)
def total_num(reviews, pos_neg):
    res = 0
    for i in range(len(reviews)):
        review = list(map(str, clean_text(reviews[i]).split(" ")))       
        review = list(filter(('').__ne__, review))
        res += len(review)

    print("The total number of words in all %s reviews: %d" % (pos_neg,res))
    return res

# apply the multi-nomial Naive Bayes classifier with Laplace smooth (Bernoulli version)
def conditional_probability(model_type, total_num, index, pos_neg, alpha):
    # formula: (the number of words in class(pos or neg) + Laplace smooth (1)) / (&total number of words in class + &bag of words size (2000))
    res = float((model_type['Count'][index] + alpha) / (total_num + (2 * alpha)))
    return res
    #CP fomular: ((# of words appearances in pos or neg) + 1) / (total # of words in pos (duplication is counted)) + 2000)
##############################

if __name__ == "__main__":
    # training set pre-processing

    # Importing the dataset
    with open("trainingSet.txt", "r") as f:
        train_sentences = []
        train_labels = []
    #read and copy every line on an array called 'lines'
        for i, line in enumerate(f):
            splitting = line.strip().split('\t')
            train_sentences.append(clean_text(splitting[0]))
            train_labels.append(int(splitting[1]))

    print(train_sentences[41])
    #print(train_labels)
    
    # Pre-processing step
    vectorizer = CountVectorizer(
    analyzer = "word"
    )
    X = vectorizer.fit_transform(train_sentences)
    #print(len(vectorizer.get_feature_names()), vectorizer.get_feature_names())
    # fit the vectorizer on the text
    vectorizer.fit(train_sentences)
    # get the vocabulary
    inv_vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
    train_vocabulary = [inv_vocab[i] for i in range(len(inv_vocab))]
    print(len(train_vocabulary))

    # generate BOW for all restaurant reviews
    print("generating train BOW of restaurant reviews ...")
    bow, name = create_bow(train_sentences, train_vocabulary)

    np_train_labels = np.array(train_labels)[np.newaxis].T
    
    bow = np.append(bow, np_train_labels, 1)
    for i in range(len(bow)):
        for j in range(len(bow[i]) - 1):
            if bow[i][j] > 0:
                bow[i][j] = 1

    # delete later
    print(bow[120])
    print(len(bow[120]))
    print(len(bow))
    counter = 0
    for i in range(len(bow[120]) - 1):
        if bow[120][i] == 1:
            counter += 1
    #

    # write pre-processed files
    np.savetxt('preprocessed_train.txt', bow, fmt="%d", delimiter=',')

    with open('preprocessed_train.txt', 'r+') as res:
        content = res.read()
        res.seek(0, 0)
        res.write(','.join(train_vocabulary) + ",class_label\n" + content)

    print(len(train_labels))
    print(counter)
    print("done ... !")

    # test set pre-processing

    # Importing the dataset
    with open("testSet.txt", "r") as t_f:
        test_sentences = []
        test_labels = []
    #read and copy every line on an array called 'lines'
        for i, line in enumerate(t_f):
            splitting = line.strip().split('\t')
            test_sentences.append(clean_text(splitting[0]))
            test_labels.append(int(splitting[1]))

    print(test_sentences[41])
    #print(train_labels)
    
    # Pre-processing step
    vectorizer = CountVectorizer(
    analyzer = "word"
    )
    X = vectorizer.fit_transform(test_sentences)
    #print(len(vectorizer.get_feature_names()), vectorizer.get_feature_names())
    # fit the vectorizer on the text
    vectorizer.fit(test_sentences)
    # get the vocabulary
    inv_vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
    test_vocabulary = [inv_vocab[i] for i in range(len(inv_vocab))]
    print(len(test_vocabulary))

    # generate BOW for all restaurant reviews
    print("generating train BOW of restaurant reviews ...")
    test_bow, name = create_bow(test_sentences, test_vocabulary)

    np_test_labels = np.array(test_labels)[np.newaxis].T
    
    test_bow = np.append(test_bow, np_test_labels, 1)
    for i in range(len(test_bow)):
        for j in range(len(test_bow[i]) - 1):
            if test_bow[i][j] > 0:
                test_bow[i][j] = 1

    # delete later
    print(test_bow[120])
    print(len(test_bow[120]))
    print(len(test_bow))
    counter = 0
    for i in range(len(test_bow[120]) - 1):
        if test_bow[120][i] == 1:
            counter += 1
    #

    # write pre-processed files
    np.savetxt('preprocessed_test.txt', test_bow, fmt="%d", delimiter=',')

    with open('preprocessed_test.txt', 'r+') as t_res:
        content = t_res.read()
        t_res.seek(0, 0)
        t_res.write(','.join(test_vocabulary) + ",class_label\n" + content)

    print(len(test_labels))
    print(counter)
    print("done ... !")

    # Classification step

    # separate reviews based on label data
    train_pos = [train_sentences[i] for i in range(len(train_labels)) if train_labels[i] == 1 ]
    train_neg = [train_sentences[i] for i in range(len(train_labels)) if train_labels[i] == 0 ]
    print(len(train_pos), len(train_neg))
    print(train_pos[0])


