#%% cell 0
import ConfigManager
import numpy as np
import tensorflow as tf
import pickle
from DatabaseManager import *
from multiprocessing import Pool
import os


#importing configurations to the application
print("Importing configurations")
confman = ConfigManager.ConfigurationManager()

#%% cell 1

# restoring dictionnaries
print("Importing word dictionnary")
with open(confman.dictionary_path, "rb") as f:
    word_dictionary = pickle.load(f)
word_dictionary_rev = dict(zip(word_dictionary.values(), word_dictionary.keys()))
print("Restoring word and doc embeddings")
word_embeddings = tf.Variable(tf.random_uniform([confman.vocabulary_size, confman.embedding_size], -1.0, 1.0))
saver = tf.train.Saver({"embeddings": word_embeddings})
sess = tf.Session()
saver.restore(sess, confman.checkpoint_path)

#%% cell 2
extracted_tweet_folder = confman.extracted_tweets
def ReadFiles(fileName):
    with open("".join([extracted_tweet_folder, "/", fileName, ".txt"]), "r", encoding="UTF-8") as f:
        Tweets = f.readline()
    return ([Tweet.strip(" \n").split(' ') for Tweet in Tweets])
userIds = np.loadtxt(confman.valid_user_ids, dtype=np.str)
threads_count = 1
with Pool(threads_count) as p:
    user_tweets = p.map(ReadFiles, userIds[:5])
