CS 331 - Spring 2020
Programming Assignment #3 - Sentiment Analysis
Team members
Junhyeok Jeong, jeongju@oregonstate.edu
Youngjoo Lee, leey3@oregonstate.edu

1. To compile the python files in this repository, you should set up virtual environment first with below commands

cd ~/CS-331
bash
virtualenv venv -p $(which python3)
source ./venv/bin/activate
pip3 install --upgrade pip
pip3 install numpy                  # to use numpy array matrix
pip3 install pandas                 # to create data templates efficiently
pip3 install sklearn                # we use sklearn for vectorize function to create bag of words efficiently

2. Check library versions with library_check.py
python3 library_check.py

- then you should see this prompts on the terminal
numpy version: 1.18.2 (or other version)
pandas version: 1.0.3 (or other version)
sklearn version: 0.22.2.post1 (or other version)

3. How to compile
-> python3 restaurant_classifier.py

# make sure to place the data files on the same directory with python file

4. check the outputs on your terminal and preprocessed_train.txt, preprocessed_test.txt, 
    and results.txt files after the input above command.

Thank you !