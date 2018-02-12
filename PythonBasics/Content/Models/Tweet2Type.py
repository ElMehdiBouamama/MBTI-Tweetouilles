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
        self.confman = ConfigManager.ConfigurationManager()
        self.dataman = DataManager.DataManager()
        self.class_tweets = self.dataman.bucketized_tweets() # Import texts + types from files
        embeddings,self.doc_embeddings = self.dataman.restore_embeddings("Variable") # Importing Embeddings and doc_embeddings
        dict,rev_dict = dataman.restore_dictionaries() # Importing dictionary and rev_dictionary

        # Initialize model variables
        self.weights = tf.Variable(tf.random_normal([self.confman.doc_embedding_size, self.confman.num_class]),dtype=tf.float32)

        # Initialize model inputs
        self.class_target = tf.placeholder(tf.int32,[None, self.confman.num_class])
        self.tweet_vectors = tf.placeholder(tf.float32,[None, self.confman.doc_embedding_size])

        pass

    def Vectorize_tweet(self,Tweets):
        """Training the model on specific tweets to get it's embedding"""
        train_doc_embedding = tf.concat(tf.random_normal([len(Tweets), self.confman.doc_embedding_size], -1, 1), self.doc_embeddings)
        
        pass

    def Fit(self):
        """Train the model to predict types based on tweets embeddings"""
        logits = tf.matmul(self.tweet_vectors, self.weights)  # matrice multiplication
        prediction = tf.nn.softmax(logits) # prediction with probabilities
        loss = tf.reduce_mean(tf.square(tf.subtract(self.class_target, prediction))) # MSE of the model
        
        optimizer = tf.train.GradientDescentOptimizer(self.logistic_learning_rate)
        optimizationStep = optimizer.minimize(loss)
        
        batch_size = self.confman.TTTbatch_size
        num_epoch = self.confman.TTTepoch

        print_loss_every = self.confman.print_loss_every
        # Train the model on a loop
        train_loss = []
        eval_loss = []
        for i in range(num_epoch):
            batch_datas, batch_labels = self.dataman.createTTTbatch(batch_size, 'Training') # Create batches from true data ([tweetVectors, target_class],[...],....)
            feed_dict = {tweet_vectors : batch_datas, class_target : batch_labels}
            
            self.sess.run(optimizationStep, feed_dict=feed_dict)
            loss_train = self.sess.run(loss, feed_dict=feed_dict)
            
            train_loss.append([i,loss_train])
            # Print loss
            if i % print_loss_every == 0:
                eval_datas, eval_labels = self.dataman.createTTTbatch(batch_size, 'Testing')
                feed_dict={tweet_vectors : eval_datas, class_targer : eval_labels}
                loss_eval = self.sess.run(loss, feed_dict=feed_dict)
                eval_loss.append([i,loss_eval])
                print('Training loss at step {} : {}'.format(i+1, loss_train))
                print('Eval loss at step {} : {}'.format(i+1, loss_eval))
        return(train_loss, eval_loss)

    def Predict(self, datas):
        """ Use a tweet vector to predict the type of personnality of the user
                datas : contains vectorized tweet vectors that should be analyzed
        """
        # use the model to predict
        logits = tf.matmul(self.tweet_vectors,self.weights)
        prediction = tf.nn.softmax(logits) # prediction 16 values of probabilities
        mostProbableClass = tf.arg_max(prediction,0) # take most probable class over all classes
        # take data and feed it to the model
        feed_dict = {tweet_vectors : datas}
        target_class = self.sess.run(mostProbableClass, feed_dict=feed_dict)
        return [self.dataman.type_rev_dict(x) for x in target_class] # return most probable class in string format

    def SaveModel(self):
        saver = tf.train.Saver({"weights": self.weights})
        save_path = saver.save(sess, self.confman.tweetToType_save_path)
        print('Model saved in file: {}'.format(save_path))
        pass

    def RestoreModel(self):
        self.weights = self.dataman.restore_weights()
        print('Model restored from file: {}'.format(self.confman.tweetToType_save_path))
        pass

    def __del__(self, **kwargs):
        self.session.close()
        pass
    


