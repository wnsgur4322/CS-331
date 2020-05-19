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
        analyzer = 'word',
        vocabulary=vocabulary
    )
    bag_of_words = vectorizer.fit_transform(input)
    bag_of_words = bag_of_words.toarray()
    
    name = vectorizer.get_feature_names()

    return bag_of_words, name


# To calculate the total number of sentences in the set (1 or 0)
def total_num(reviews, pos_neg):
    res = 0
    for i in range(len(reviews)):
        review = list(map(str, clean_text(reviews[i]).split(" ")))       
        review = list(filter(('').__ne__, review))
        res += len(review)

    print("The total number of words in all %s reviews: %d" % (pos_neg,res))
    return res

# apply the Naive Bayes classifier with Laplace smooth (Bernoulli version)
def conditional_probability(model_type, total_num, index, pos_neg, alpha):
    # formula: (the number of sentences with word i in class(1 or 0) + Laplace smooth (1)) / (&total number of sentences in class + 2*Laplace smooth)
    res = float((model_type['Count'][index] + alpha) / (total_num + (2 * alpha)))
    return res


from functools import reduce
def prediction(sentence, pos_CP, neg_CP, train_pos_prob, train_neg_prob):
        word_count = []

        for j in range(len(train_model['Word'])):
            word_count.append(sentence.count(train_model['Word'][j]))

        pos_pow_list = [math.log(wi) * n for wi, n in zip(pos_CP, word_count)]
        pos_pow_list = list(filter((0.0).__ne__, pos_pow_list))
        

        neg_pow_list = [math.log(wi) * n for wi, n in zip(neg_CP, word_count)]
        neg_pow_list = list(filter((0.0).__ne__, neg_pow_list))
        
        # P(1 | sets) = train_pos_prob * pos_CP[word_1]^n * pos_CP[word_2] ....
        pos_res = math.log(train_pos_prob) + reduce(lambda x, y: x + y, pos_pow_list)
        
        # P(0 | sets) = train_neg_prob * neg_CP[word_1]^n * neg_CP[word_2] ....
        neg_res = math.log(train_neg_prob) + reduce(lambda x, y: x + y, neg_pow_list)

        if pos_res > neg_res:
            # 1
            return("1")
        else:
            # 0
            return("0") 


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
    
    # Pre-processing step
    vectorizer = CountVectorizer(
    analyzer = "word"
    )
    X = vectorizer.fit_transform(train_sentences)
    # fit the vectorizer on the text
    vectorizer.fit(train_sentences)
    # get the vocabulary
    inv_vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
    train_vocabulary = [inv_vocab[i] for i in range(len(inv_vocab))]

    # generate BOW for all restaurant reviews
    print("\ngenerating train BOW of restaurant reviews ...")
    bow, name = create_bow(train_sentences, train_vocabulary)

    np_train_labels = np.array(train_labels)[np.newaxis].T
    
    bow = np.append(bow, np_train_labels, 1)
    for i in range(len(bow)):
        for j in range(len(bow[i]) - 1):
            if bow[i][j] > 0:
                bow[i][j] = 1

    # write pre-processed files
    np.savetxt('preprocessed_train.txt', bow, fmt="%d", delimiter=',')

    with open('preprocessed_train.txt', 'r+') as res:
        content = res.read()
        res.seek(0, 0)
        res.write(','.join(train_vocabulary) + ",class_label\n" + content)

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

    

    # generate BOW for all restaurant reviews
    print("\ngenerating train BOW of restaurant reviews ...")
    test_bow, test_name = create_bow(test_sentences, train_vocabulary)

    np_test_labels = np.array(test_labels)[np.newaxis].T
    
    test_bow = np.append(test_bow, np_test_labels, 1)
    for i in range(len(test_bow)):
        for j in range(len(test_bow[i]) - 1):
            if test_bow[i][j] > 0:
                test_bow[i][j] = 1
    print("done ... !")
    
    
    # write pre-processed files
    print("\nWriting pre-processed files...")
    np.savetxt('preprocessed_test.txt', test_bow, fmt="%d", delimiter=',')

    with open('preprocessed_test.txt', 'r+') as t_res:
        content = t_res.read()
        t_res.seek(0, 0)
        t_res.write(','.join(train_vocabulary) + ",class_label\n" + content)
    
    print("done ... !")

    '''
    # Classification step

    # separate training reviews based on label data
    train_pos = [train_sentences[i] for i in range(len(train_labels)) if train_labels[i] == 1 ]
    train_neg = [train_sentences[i] for i in range(len(train_labels)) if train_labels[i] == 0 ]

    # Make train models
    print("\nMaking training models...")
    bows = np.sum(train_sentences, axis=0)

    train_model = pd.DataFrame( 
    (count, word) for word, count in
    zip(bows, name))
    train_model.columns = ['Word', 'Count']
    print("done ... !")

    # Priors: training set's (1) and (0) probabilities
    print("\nCaculating positive(1) and negative(0) prbabilities...")
    train_pos_prob = len(train_pos) / len(train_sentences)
    train_neg_prob = len(train_neg) / len(train_sentences)
    print("probabilities of training set\npositive(1): %f\nnegative(0): %f" % (train_pos_prob, train_neg_prob))
    print("done ... !")


    # Learns the parameters used by the classifier
    print("\nLearning the parameters used by the classifier...")
    posbow_train, posname_train = create_bow(train_pos, train_vocabulary)
    posbow_train = np.sum(posbow_train, axis=0)


    postrain_model = pd.DataFrame( 
    (count, word) for word, count in
    zip(posbow_train, posname_train))
    postrain_model.columns = ['Word', 'Count']
    
    negbow_train, negname_train = create_bow(train_neg, train_vocabulary)
    negbow_train = np.sum(negbow_train, axis=0)

    negtrain_model = pd.DataFrame( 
    (count, word) for word, count in
    zip(negbow_train, negname_train))
    negtrain_model.columns = ['Word', 'Count']

    #P(Wi...len(train_vocabulary)|1) Part
    pos_CP = []
    total_pos = total_num(train_pos, 1)
    for i in range(len(postrain_model)):
        pos_CP.append(conditional_probability(postrain_model, total_pos, i, 1, 1))

    #P(Wi...len(train_vocabulary)|0) Part
    neg_CP = []
    total_neg = total_num(train_neg, 0)
    for i in range(len(negtrain_model)):
        neg_CP.append(conditional_probability(negtrain_model, total_neg, i, 0, 1))
    print("done ... !")


    # Calculate training accuracy of the Naive Bayes classifier (training)
    print("\n-----Calculating training accuracy of the Naive Bayes Classifier-----")
    train_predictions = []

    for i in range(len(train_sentences)):
        train_predictions.append(prediction(train_sentences[i], pos_CP, neg_CP, train_pos_prob, train_neg_prob))

    train_accuracy = 0
    for i in range(len(train_predictions)):
        if int(train_predictions[i]) == train_labels[i]:
            train_accuracy += 1
    print("--The training accuracy of the Naive Bayes classifier: %f" % float(train_accuracy/len(train_predictions)))


    # Calculate test accuracy of the Naive Bayes classifier (test)
    print("\n-----Calculating test accuracy of the Naive Bayes Classifier-----")
    test_predictions = []

    for i in range(len(test_sentences)):
        test_predictions.append(prediction(test_sentences[i], pos_CP, neg_CP, train_pos_prob, train_neg_prob))

    test_accuracy = 0
    for i in range(len(test_predictions)):
        if int(test_predictions[i]) == test_labels[i]:
            test_accuracy += 1

    print("--The test accuracy of the Naive Bayes classifier: %f" % float(test_accuracy/len(test_predictions)))
    '''

    # Classification step
    bows = []
    for i in range(len(train_vocabulary)):
        bows.append(sum(row[i] for row in bow[:]))
    
    # separate training reviews based on label data
    train_pos = [0]*len(train_vocabulary)
    train_neg = [0]*len(train_vocabulary)

    for i in range(len(bow)):
        for j in range(len(train_vocabulary)):
            if (bow[i][-1] == 1):
                train_pos[j] += bow[i][j]
            else:
                train_neg[j] += bow[i][j]

    # Make train models
    print("\nMaking training models...")

    train_model = pd.DataFrame( 
    (count, word) for word, count in
    zip(bows, name))
    train_model.columns = ['Word', 'Count']
    print("done ... !")

    # Priors: training set's (1) and (0) probabilities
    print("\nCaculating positive(1) and negative(0) prbabilities...")
    train_pos_prob = sum(train_pos) / sum(bows)
    train_neg_prob = sum(train_neg) / sum(bows)
    print("probabilities of training set\npositive(1): %f\nnegative(0): %f" % (train_pos_prob, train_neg_prob))
    print("done ... !")

    # Learns the parameters used by the classifier
    print("\nLearning the parameters used by the classifier...")

    #pos and neg models
    postrain_model = pd.DataFrame( 
    (count, word) for word, count in
    zip(train_pos, name))
    postrain_model.columns = ['Word', 'Count']


    negtrain_model = pd.DataFrame( 
    (count, word) for word, count in
    zip(train_neg, name))
    negtrain_model.columns = ['Word', 'Count']


    #P(Wi...len(train_vocabulary)|1) Part
    pos_CP = []
    total_pos = sum(train_pos)
    for i in range(len(postrain_model)):
        pos_CP.append(conditional_probability(postrain_model, total_pos, i, 1, 1))

    #P(Wi...len(train_vocabulary)|0) Part
    neg_CP = []
    total_neg = sum(train_neg)
    for i in range(len(negtrain_model)):
        neg_CP.append(conditional_probability(negtrain_model, total_neg, i, 0, 1))
    print("done ... !")


    # Calculate training accuracy of the Naive Bayes classifier (training)
    print("\n-----Calculating training accuracy of the Naive Bayes Classifier-----")
    train_predictions = []

    for i in range(len(bow)):
        train_predictions.append(prediction(train_sentences[i], pos_CP, neg_CP, train_pos_prob, train_neg_prob))

    train_accuracy = 0
    for i in range(len(train_predictions)):
        if int(train_predictions[i]) == train_labels[i]:
            train_accuracy += 1
    print("--The training accuracy of the Naive Bayes classifier: %f" % float(train_accuracy/len(train_predictions)))


    # Calculate test accuracy of the Naive Bayes classifier (test)
    print("\n-----Calculating test accuracy of the Naive Bayes Classifier-----")
    test_predictions = []

    for i in range(len(test_sentences)):
        test_predictions.append(prediction(test_sentences[i], pos_CP, neg_CP, train_pos_prob, train_neg_prob))

    test_accuracy = 0
    for i in range(len(test_predictions)):
        if int(test_predictions[i]) == test_labels[i]:
            test_accuracy += 1

    print("--The test accuracy of the Naive Bayes classifier: %f" % float(test_accuracy/len(test_predictions)))


