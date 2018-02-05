import sys
import numpy as np
from DatabaseManager import *
import multiprocessing
from multiprocessing import Pool
from text_helper import *
import DataManager
import ConfigManager
import random
import collections
import os
import pandas as pd
import tensorflow as tf


class Tweet2Type(object):
    """Use this class to predict personnality of people from they're tweets"""
    def __init__(self, path):
        """The path represents the config file path"""
        confman = ConfigManager.ConfigurationManager()
        dataman = DataManager.DataManager()
        self.class_tweets = dataman.getBucketizedTweets() # Should replace this with usertweets + userType
        self.embeddings,self.doc_embeddings = dataman.restoreEmbeddings("Constant") # Importing Embeddings and doc_embeddings
        dict,rev_dict = dataman.restoreDictionaries() # Importing dictionary and rev_dictionary

        # Initialize model variables
        self.weights = tf.Variable(tf.random_normal([confman.doc_embedding_size, confman.num_class]),dtype=tf.float32)

        # Initialize model inputs
        self.class_target = tf.placeholder(tf.int32,[None, confman.num_class])
        self.tweet_vectors = tf.placeholder(tf.float32,[None, confman.doc_embedding_size])

        pass

    def Vectorize_tweet(self,tweet):
        """Training the model on specific tweets to get it's embedding"""
        pass

    def Fit(self):
        """Train the model to predict types based on tweets embeddings"""
        logits = tf.matmul(tweet_vectors, weights) 
        prediction = tf.nn.softmax(logits)
        loss = tf.reduce_mean(tf.square(tf.subtract(class_target, prediction)))
        
        optimizer = tf.train.GradientDescentOptimizer(logistic_leadrning_rate)
        optimizationStep = optimizer.minimize(loss)
        
        batch_size = self.confman.TTTbatch_size
        num_epoch = self.confman.TTTepoch
        # Train the model on a loop
        for i in range(num_epoch):
            batch_data = self.dataman.createTTTbatch(batch_size,class_tweets) # Create batches from true data ([tweetVectors, target_class],[...],....)
        
        pass

    def Predict(self):
        """ Use a tweet vector to predict the type of personnality of the user"""
        logits = tf.matmul(tweet_vectors,weights)
        prediction = tf.nn.log_softmax(logits)
        mostProbableClass = tf.arg_max(prediction,0)
        return mostProbableClass

    def SaveModel(self):
        pass

    def __del__(self, **kwargs):
        self.session.close()
        pass
    


