#%% cell 0
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
    def __init__(self):
        """The path represents the config file path"""
        # Initialize class variables and import configuration manager and data manager with respective items
        self.confman = ConfigManager.ConfigurationManager()
        self.dataman = DataManager.DataManager()
        self.class_tweets = self.dataman.bucketized_tweets() # Import texts + types from files
        self.embeddings,self.doc_embeddings = self.dataman.restore_embeddings("Constant") # Importing Embeddings and doc_embeddings
        self.dict,self.rev_dict = self.dataman.restore_dictionaries() # Importing dictionary and rev_dictionary
        self.logistic_learning_rate = self.confman.logistic_learning_rate
        # Initialize model variables
        self.weights = tf.Variable(tf.random_normal([self.confman.doc_embedding_size, self.confman.num_class]),dtype=tf.float32)

        # Initialize model inputs
        self.class_target = tf.placeholder(tf.int32,[None, self.confman.num_class])
        self.tweet_vectors = tf.placeholder(tf.float32,[None, self.confman.doc_embedding_size])

        # Initialize tensorflow session
        self.sess = tf.Session()
        pass
    
    """Training the model on specific tweets to get it's embedding
       text_tweets : array of tweet text that should be given in the format ['TweeOne', 'TweetTwo', ...] """
    def Vectorize_tweet(self,text_tweets):
        # Clean tweets from unecessary tookens
        tweets = normalize_text(text_tweets)
        number_tweets = text_to_numbers(tweets, self.dict)

        # Intialize model variables
        batch_size = len(Tweets) * 2 
        window_size = self.confman.window_size 
        num_epoch = self.confman.DTVnum_epoch
        print_loss_every = self.confman.print_loss_every
        num_sampled = self.confman.num_sampled
        vocabulary_size = self.confman.vocabulary_size
        num_sampled = int(batch_size/2)
        doc_embedding_size = self.confman.doc_embedding_size
        model_learning_rate = self.confman.model_learning_rate

        # NCE loss parameters and tweet embeddings
        nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, concatenated_size], stddev=1.0 / np.sqrt(concatenated_size)))
        nce_biases = tf.Variable(tf.random_normal([vocabulary_size]))
        doc_embeddings = tf.Variable(tf.random_uniform([len(text_tweets), doc_embedding_size], -1.0, 1.0))

        # Create data/target placeholders
        x_inputs = tf.placeholder(tf.int32, shape=[None, window_size + 1]) # plus 1 for doc index
        y_target = tf.placeholder(tf.int32, shape=[None, 1])
        
        # Lookup the word embedding
        # Add together element embeddings in window:
        embed = tf.zeros([batch_size, embedding_size])
        for element in range(window_size):
            embed += tf.nn.embedding_lookup(self.embeddings, x_inputs[:, element])

        doc_indices = tf.slice(x_inputs, [0, window_size], [batch_size, 1])
        doc_embed = tf.nn.embedding_lookup(doc_embeddings, doc_indices)

        # Concatenate embeddings
        final_embed = tf.concat([embed, tf.squeeze(doc_embed)], 1)

        # Get loss from prediction
        loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, y_target, final_embed, num_sampled, vocabulary_size))

        # Create optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=model_learning_rate)
        train_step = optimizer.minimize(loss)
                
        #Add variable initializer
        self.sess.run(tf.global_variables_initializer())

        # Run the skip gram model.
        loss_vec = []
        loss_x_vec = []
        print('Training tweet vectors for the {} given tweets'.format(len(text_tweets)))
        for i in num_epoch:
            batch_inputs, batch_labels = self.dataman.create_wtv_batch(number_tweets, batch_size, window_size) # create batches
            feed_dict = {x_inputs : batch_inputs, y_target : batch_labels}

            # Run the train step
            self.sess.run(train_step, feed_dict=feed_dict)

            # Return the loss
            if (i+1) % print_loss_every == 0:
                loss_val = sess.run(loss, feed_dict=feed_dict)
                loss_vec.append(loss_val)
                loss_x_vec.append(i+1)
                print('Loss at step {} : {}'.format(i+1, loss_val))

        # Convert tensors to numpy arrays
        tweet_embeddings = self.sess.run(doc_embeddings)
        # Return embeddings
        return tweet_embeddings 

    """Train the model to fit (tweet vector/type) embedding space mapping """
    def Fit(self):
        print("Initializing Model Variables")

        logits = tf.matmul(self.tweet_vectors, self.weights)  # matrice multiplication
        prediction = tf.nn.softmax(logits) # prediction with probabilities
        loss = tf.reduce_mean(tf.square(tf.subtract(tf.cast(self.class_target, tf.float32), prediction))) # MSE of the model
        
        optimizer = tf.train.GradientDescentOptimizer(self.logistic_learning_rate)
        optimizationStep = optimizer.minimize(loss)
        
        batch_size = self.confman.TTTbatch_size
        num_epoch = self.confman.TTTepoch
        print_loss_every = self.confman.print_loss_every

        self.sess.run(tf.global_variables_initializer())

        print("Printing 5 batch examples")
        batch_datas, batch_labels = self.dataman.create_ttt_batch(batch_size, self.embeddings, self.doc_embeddings, 'Training') # Batch examples
        
        print(np.shape(batch_datas))
        print(np.shape(batch_labels))

        # Train the model on a loop
        print("Starting training")
        train_loss = []
        eval_loss = []
        for i in range(num_epoch):
            batch_datas, batch_labels = self.dataman.create_ttt_batch(batch_size, self.embeddings, self.doc_embeddings, 'Training') # Create batches from true data ([tweetVectors, target_class],[...],....)
            feed_dict = {self.tweet_vectors : batch_datas, self.class_target : batch_labels}
            
            self.sess.run(optimizationStep, feed_dict=feed_dict)
            loss_train = self.sess.run(loss, feed_dict=feed_dict)
            
            train_loss.append([i+1, loss_train])
            # Print loss
            if i % print_loss_every == 0:
                eval_datas, eval_labels = self.dataman.create_ttt_batch(batch_size, self.embeddings, self.doc_embeddings, 'Testing')
                feed_dict={self.tweet_vectors : eval_datas, self.class_target : eval_labels}
                loss_eval = self.sess.run(loss, feed_dict=feed_dict)
                eval_loss.append([i+1, loss_eval])
                print('Training loss at step {} : {}'.format(i+1, loss_train))
                print('Eval loss at step {} : {}'.format(i+1, loss_eval))
        return(train_loss, eval_loss)
    
    """ Use a tweet vector to predict the type of personnality of the user
                datas : contains vectorized tweet vectors that should be analyzed """
    def Predict(self, datas):
        # use the model to predict
        logits = tf.matmul(self.tweet_vectors,self.weights)
        prediction = tf.nn.softmax(logits) # prediction 16 values of probabilities
        mostProbableClass = tf.arg_max(prediction,0) # take most probable class over all classes

        # take data and feed it to the model
        feed_dict = {self.tweet_vectors : datas}
        target_class = self.sess.run(mostProbableClass, feed_dict=feed_dict)
        # return most probable class in string format
        return [self.dataman.type_rev_dict(x) for x in target_class] 

    """ Save the model parameters in a file """
    def SaveModel(self):
        saver = tf.train.Saver({"weights": self.weights})
        save_path = saver.save(self.sess, self.confman.tweetToType_save_path)
        print('Model saved in file: {}'.format(save_path))
        pass

    """ Save the model parameters from the file """
    def RestoreModel(self):
        self.weights = self.dataman.restore_weights()
        print('Model restored from file: {}'.format(self.confman.tweetToType_save_path))
        pass

    """ destroy related objects """
    def __del__(self, **kwargs):
        self.sess.close()
        return self.super().__del__()
    


