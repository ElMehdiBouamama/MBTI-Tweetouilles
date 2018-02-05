#%% cell 0
from functools import Partial
import sys
import os
import ConfigManager
import multiprocessing
from multiprocessing import Pool
from DatabaseManager import *
import collections
import pickle
import numpy as np
import tensorflow as tf
from text_helper import *

extracted_tweet_folder = " "
def ReadFiles(fileName):
    with open(extracted_tweet_folder + "/" + fileName + ".txt", "r", encoding="UTF-8") as f:
        Tweets = f.readlines()
    return [Tweet.strip(" \n").split(' ') for Tweet in Tweets]

def ReadClassifiedFiles(fileName,datas):
    with open(extracted_tweet_folder + "/" + fileName + ".txt", "r", encoding="UTF-8") as f:
        Tweets = f.readlines()
    return ([Tweet.strip(" \n").split(' ') for Tweet in Tweets], GetMbtiOfUser(datas,fileName))

class DataManager(object):
    def __init__(self):
        ''' Upload tweets from files and get them ready to use by tweet2vec and tweet2type classes '''
        self.configuration_manager = ConfigManager.ConfigurationManager() # Initialize configuration data
        print('Importing json data')
        self.tweet_datas = ReadJsonFile(self.configuration_manager.tweets_json) # Importing json file with all informations about users
        print('Importing valid user ids')
        self.userIds = np.loadtxt(self.configuration_manager.valid_user_ids, dtype=np.str) # Importing valid user id's to check who are the users that really have tweets
        self.sess = tf.Session()

    # Get all tweets of each user
    def getUserTweets(self):
        if(self.users_tweets is None):
            threads_count = os.cpu_count()*6 # Number of thread per CPU
            global extracted_tweet_folder
            extracted_tweet_folder = self.configuration_manager.extracted_tweets # Get tweets folder from configuration manager
            # Each CPU will have 6 threads to manage at every timestep
            with Pool(threads_count) as p:
                self.users_tweets = p.map(ReadFiles,self.userIds) # Asynchronously extract tweeter texts
        return self.user_tweets

    # Get all tweets with the classes of each user
    def getBucketizedTweets(self):
        if(self.class_tweets is None):
            threads_count = os.cpu_count()*6
            global extracted_tweet_folder
            extracted_tweet_folder = self.configuration_manager.extracted_tweets
            readFiles = Partial(ReadClassifiedFiles, datas=self.tweet_datas)
            with Pool(threads_count) as p:
                self.class_tweets = p.map(readFiles,self.userIds)
        return self.class_tweets

    # Get tweets without user separation
    def getAllTweets(self): 
        if(self.texts is None):
            self.texts = []
            for user_tweets in users_tweets:
                for tweet in user_tweets:
                    self.texts.append(tweet)
        return self.texts

    def restoreEmbeddings(self,tensorType="Variable"):
        if(tensorType == "Constant"):
            embeddings = tf.constant(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), tf.float32)
            doc_embeddings = tf.constant(tf.random_uniform([number_of_tweets, doc_embedding_size], -1.0, 1.0), tf.float32)
        else:
            embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            doc_embeddings = tf.Variable(tf.random_uniform([number_of_tweets, doc_embedding_size], -1.0, 1.0))
        saver = tf.train.Saver({"embeddings": embeddings, "doc_embeddings": doc_embeddings}) # Import Embeddings
        saver.restore(self.sess, self.configuration_manager.checkpoint_path) # Restore data from checkpoint path
        return(embeddings, doc_embeddings)

    def restoreDictionaries(self):
        if(self.word_dictionary is None):
            print('Dictionary Backup')
            with open(self.configuration_manager.dictionary_path),"rb") as f:
                self.word_dictionary = pickle.load(f)
            self.word_dictionary_rev = dict(zip(self.word_dictionary.values(), self.word_dictionary.keys())) # Import dictionary
        return(self.word_dictionary, self.word_dictionary_rev)
    
    # Generate data randomly (N words behind, target, N words ahead)
    def createTTTbatch(batch_size,bucketized_tweets):
        
        pass