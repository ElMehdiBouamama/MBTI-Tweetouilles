#%% cell 0
import sys
import os
from functools import partial
import ConfigManager
import multiprocessing
from multiprocessing import Pool
from DatabaseManager import *
import collections
import pickle
import numpy as np
import tensorflow as tf
from text_helper import *
import pandas as pd

extracted_tweet_folder = " "
def ReadFiles(fileName):
    with open(extracted_tweet_folder + "/" + fileName + ".txt", "r", encoding="UTF-8") as f:
        Tweets = f.readlines()
    return [Tweet.strip(" \n").split(' ') for Tweet in Tweets]

def read_classified_files(fileName,datas):
    file_path = extracted_tweet_folder + "/" + fileName + ".txt"
    if(os.path.exists(file_path)):
        with open(extracted_tweet_folder + "/" + fileName + ".txt", "r", encoding="UTF-8") as f:
            Tweets = f.readlines()
        return ([Tweet.strip(" \n").split(' ') for Tweet in Tweets], GetMbtiOfUser(datas,fileName))
    else:
        return ["",None]

class DataManager(object):
    def __init__(self):
        ''' Upload tweets from files and get them ready to use by tweet2vec and tweet2type classes '''
        self.sess = tf.Session()
        self.configuration_manager = ConfigManager.ConfigurationManager() # Initialize configuration data
        print('Importing json data')
        all_users = ReadJsonFile(self.configuration_manager.tweets_json) # Importing json file with all informations about users
        print('Importing valid user ids')
        self.userIds = np.loadtxt(self.configuration_manager.valid_user_ids, dtype=np.str) # Importing valid user id's to check who are the users that really have tweets
        # clean json data from unwanted users
        print('Spliting data into 3 sets')
        self.tweet_datas = dict()
        for x in all_users:
            if x in self.userIds:
                self.tweet_datas.update(dict({x:all_users[x]}))
        # split data into train / test / validation
        self.train_data = np.random.choice(self.userIds, size=int(len(self.userIds)*0.6))
        # test userIds
        testing_data = dict()
        for x in self.userIds:
            if x not in self.train_data:
                testing_data.update(dict({x:all_users[x]}))
        self.test_data = np.random.choice(list(testing_data.keys()), size=int(len(testing_data)*0.5))
        # validation userIds
        self.valid_data = dict()
        for x in testing_data:
            if x not in self.test_data:
                self.valid_data.update(dict({x:all_users[x]}))
        # initialize global variables and dictionaries
        print('Creating type dictionnary')
        self.word_dictionary = []
        self.type_dict = {'ENFJ':0, 'INFJ':1, 'INTJ':2, 'ENTJ':3, 'ENTP':4, 'INTP':5 ,'INFP':6, 'ENFP':7, 'ESFP':8, 'ISFP':9, 'ISTP':10, 'ESTP':11, 'ESFJ':12, 'ISFJ':13, 'ISTJ':14, 'ESTJ':15}
        self.type_rev_dict = dict(zip(self.type_dict.values(), self.type_dict.keys()))
        self.class_tweets = None
        print('Importing cumulative tweet count array')
        self.cum_tweet_array = [0,*(np.cumsum(GetCountArrayOfConfirmedTweet(self.tweet_datas)))]
        print('Creating id to ix dictionnary')
        ids = self.tweet_datas.keys()
        self.id_to_ix = dict()
        for ix,id in enumerate(ids):
            self.id_to_ix.update(dict({id:ix}))
        pass
    
    
    # Get all tweets of each user
    def user_tweets(self):
        if(self.users_tweets is None):
            threads_count = os.cpu_count()*6 # Number of thread per CPU
            global extracted_tweet_folder
            extracted_tweet_folder = self.configuration_manager.extracted_tweets # Get tweets folder from configuration manager
            # Each CPU will have 6 threads to manage at every timestep
            with Pool(threads_count) as p:
                self.users_tweets = p.map(ReadFiles,self.userIds) # Asynchronously extract tweeter texts
        return self.user_tweets



    # Get all tweets with the classes of each user
    def bucketized_tweets(self):
        if(self.class_tweets is None):
            threads_count = os.cpu_count()*6
            global extracted_tweet_folder
            extracted_tweet_folder = self.configuration_manager.extracted_tweets
            readFiles = partial(read_classified_files, datas=self.tweet_datas) # Proxy function of read files with data included as parameter
            with Pool(threads_count) as p:
                self.class_tweets = p.map(readFiles,self.userIds) # Asynchronously extract text with type associated
        return self.class_tweets
    
    # Get tweets without user separation
    def all_tweets(self): 
        if(self.texts is None):
            self.texts = []
            for user_tweets in users_tweets:
                for tweet in user_tweets:
                    self.texts.append(tweet)
        return self.texts

    # restore embeddings in Variable or Constant tensors
    def restore_embeddings(self,tensorType="Variable"):
        embeddings = tf.Variable(tf.random_uniform([self.configuration_manager.vocabulary_size, self.configuration_manager.embedding_size], -1.0, 1.0))
        doc_embeddings = tf.Variable(tf.random_uniform([self.configuration_manager.number_of_tweets, self.configuration_manager.doc_embedding_size], -1.0, 1.0))
        saver = tf.train.Saver({"embeddings": embeddings, "doc_embeddings": doc_embeddings}) # Import Embeddings
        saver.restore(self.sess, self.configuration_manager.checkpoint_path) # Restore data from checkpoint path
        if(tensorType=="Constant"):
            embeddings = self.sess.run(embeddings)
            doc_embeddings = self.sess.run(doc_embeddings)
        return(embeddings, doc_embeddings)
    # restore Tweet2Type weights
    def restore_weights(self):
        weights = tf.Variable(tf.random_normal([self.configuration_manager.doc_embedding_size, self.configuration_manager.num_class]),dtype=tf.float32)
        saver = tf.train.Saver({"weights": weights})
        saver.restore(self.sess, self.configuration_manager.tweetToType_save_path)
        return weights


    #restore dictionnaries
    def restore_dictionaries(self):
        if len(self.word_dictionary) == 0:
            print('Dictionary Backup')
            with open(self.configuration_manager.dictionary_path,"rb") as f:
                self.word_dictionary = pickle.load(f)
            self.word_dictionary_rev = dict(zip(self.word_dictionary.values(), self.word_dictionary.keys())) # Import dictionary
        return(self.word_dictionary, self.word_dictionary_rev)

    # Generate data randomly (N words behind, target, N words ahead)
    def create_ttt_batch(self, batch_size, embeddings, doc_embeddings, batch_type="Training"):
        # batch_data contains vectorized tweets
        batch_data = []
        # label_data contains mbti types from 0 to 15
        label_data = []
        # initialize data with the type of data we want to generate
        data = None
        if batch_type == 'Validation':
            data = self.valid_data
            batch_size = len(doc_embeddings)*0.2
        elif batch_type == 'Testing':
            data = self.test_data
            batch_size = len(doc_embeddings)*0.2
        else:
            data = self.train_data
        while len(batch_data) < batch_size:
            # select random user to start
            user_id_ix = np.random.randint(len(data))
            rand_user_ix = int(data[user_id_ix])
            # Checking if user is a valid user before continue
            if(str(rand_user_ix) not in self.userIds):
                continue
            userTweetCount = GetCountOfConfirmedTweetOfUser(self.tweet_datas, str(rand_user_ix))
            if(userTweetCount==0):
                continue
            # select a random tweet from user tweets
            rand_tweet_ix = np.random.randint(userTweetCount)  
            # select doc embedding
            doc_ix = self.cum_tweet_array[self.id_to_ix[str(rand_user_ix)]] + rand_tweet_ix # select doc embedding index using user_ix and tweet_ix
            doc_ix = doc_ix % self.configuration_manager.number_of_tweets
            batch_data.append(doc_embeddings[doc_ix]) # Extract doc_embedding from specific user
            # get user labels and bucketize them
            user_type = GetMbtiOfUser(self.tweet_datas, str(rand_user_ix))
            x = self.type_dict[user_type]
            bucketized_tweet = np.zeros(self.configuration_manager.num_class) # create array for the array
            bucketized_tweet[x] = 1 # add one at the correct position of the class
            label_data.append(bucketized_tweet) # add the array to batch
        # Convert batch_data to np array
        batch_data = np.array(batch_data)
        label_data = np.array(label_data)
        return(batch_data, label_data)

    # Generate data randomly (N words behind, target, N words ahead)
    def create_wtv_batch(self, sentences, batch_size, window_size):
        # Fill up data batch
        batch_data = []
        label_data = []
        while len(batch_data) < batch_size:
            # select random sentence to start
            rand_sentence_ix = int(np.random.choice(len(sentences), size=1))
            rand_sentence = sentences[rand_sentence_ix]
            # Generate consecutive windows to look at
            window_sequences = [rand_sentence[max((ix-window_size),0):(ix+window_size+1)] for ix, x in enumerate(rand_sentence)]
            # Denote which element of each window is the center word of interest
            label_indices = [ix if ix<window_size else window_size for ix,x in enumerate(window_sequences)]
        
            # Pull out center word of interest for each window and create a tuple for each window
            # For doc2vec we keep LHS window only to predict target word
            batch_and_labels = [(rand_sentence[i:i+window_size], rand_sentence[i+window_size]) for i in range(0, len(rand_sentence)-window_size)]
            if(len(batch_and_labels) < 2 ):
                continue
            batch, labels = [list(x) for x in zip(*batch_and_labels)]
            # Add document index to batch!! Remember that we must extract the last index in batch for the doc-index
            batch = [x + [rand_sentence_ix] for x in batch]
            # extract batch and labels
            batch_data.extend(batch[:batch_size])
            label_data.extend(labels[:batch_size])
        # Trim batch and label at the end
        batch_data = batch_data[:batch_size]
        label_data = label_data[:batch_size]
        # Convert to numpy array
        batch_data = np.array(batch_data)
        label_data = np.transpose(np.array([label_data]))
    
        return(batch_data, label_data)
