# CS 331 - Spring 2020
# Programming Assignment 3 - Sentiment Analysis
# Junhyeok Jeong, jeongju@oregonstate.edu
# Youngjoo Lee, leey3@oregonstate.edu

import math

if __name__ == "__main__":
    # Importing the dataset
    with open("trainingSet.txt", "r") as f:
        sentences = []
        labels = []
    #read and copy every line on an array called 'lines'
        for i, line in enumerate(f):
            split = line.strip().split('\t')
            sentences.append(split[0])
            labels.append(int(split[1]))
    print(sentences)
    print(labels)
    