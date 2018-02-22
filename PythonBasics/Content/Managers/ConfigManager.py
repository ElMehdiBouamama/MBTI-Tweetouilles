#%% cell 0
import configparser
import os
import sys


class ConfigurationManager(object):
    def __init__(self):
        ''' Create a config parser and get general configuration parameters from config file '''
        self.config = configparser.ConfigParser() # Create ConfigParser to access configuration informations
        self.config.read(self.getConfigPath()) # Read informations from the config file
        self.config.set('PATHS','project_folder',self.getProjectPath())
        # Get all global variables from tweet2vec file
        self.vocabulary_size = self.config['DocToVec'].getint('vocabulary_size')
        self.embedding_size = self.config['DocToVec'].getint('embedding_size')
        self.number_of_tweets = self.config['DocToVec'].getint('number_of_tweets')
        self.doc_embedding_size = self.config['DocToVec'].getint('doc_embedding_size')
        self.window_size = self.config['DocToVec'].getint('window_size')
        self.DTVbatch_size = self.config['DocToVec'].getint('batch_size')
        self.DTVnum_epoch = self.config['DocToVec'].getint('generations')

        self.num_class = self.config['TweetToType'].getint('num_class')
        self.TTTbatch_size = self.config['TweetToType'].getint('batch_size')
        self.logistic_learning_rate = self.config['TweetToType'].getfloat('model_learning_rate')
        self.print_loss_every = self.config['TweetToType'].getint('print_loss_every')
        self.TTTepoch = self.config['TweetToType'].getint('generations')

        self.valid_user_ids = self.config['PATHS']['valid_user_ids']
        self.tweets_json = self.config['PATHS']['tweets_json']
        self.checkpoint_path = self.config['PATHS']['doc2vec_save_path']
        self.dictionary_path = self.config['PATHS']['dictionary_path']
        self.extracted_tweets = self.config['PATHS']['extracted_tweets']
        self.tweetToType_save_path = self.config['PATHS']['tweet2type_save_path']
        
        pass

    def getProjectPath(self):
        if(os.getcwd() == 'C:\\Users\\BOUÂMAMAElMehdi\\documents\\visual studio 2017\\Projects\\PythonBasics\\PythonBasics'):
            return 'C:/Users/BOUÂMAMAElMehdi/documents/visual studio 2017/Projects/PythonBasics/PythonBasics'
        else:
            return os.getcwd()

    def getConfigPath(self):
        return self.getProjectPath() + "/Configs/tweet2vec.ini"

    def __str__(self):
        content = "The configuration file contains those elements :\n"
        for i in config:
            content = content + "----- " + i + " Section -----\n"
            for j in config[i]:
                content = content + j + "\n"
        return content

    
