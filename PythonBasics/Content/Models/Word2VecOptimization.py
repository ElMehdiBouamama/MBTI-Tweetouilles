#%% cell 0
import sys
import numpy as np
from DatabaseManager import *
import multiprocessing
from multiprocessing import Pool
from text_helper import *
import random
import collections
import os
import pandas as pd
import tensorflow as tf
import pickle


project_folder = "./"
save_data_folder = project_folder + "SaveFolder/"
extracted_tweet_folder = project_folder + "ExtractedTweets/"
datas = ReadJsonFile(project_folder + "TwiSty-FR.json")
userIds = np.loadtxt(project_folder + 'ValidUserIds.txt' ,dtype=np.str)

def ReadFiles(fileName):
    tweet_sentences = []
    with open(extracted_tweet_folder + fileName + ".txt", "r", encoding="UTF-8") as f:
        Tweets = f.readlines()
    return [Tweet.strip(" \n").split(' ') for Tweet in Tweets]

#%% cell 1
with Pool(150) as p:
    users_tweets = p.map(ReadFiles,userIds)

texts = []
for user_tweets in users_tweets:
    for tweet in user_tweets:
        texts.append(tweet)

print(texts[:2])

sess = tf.Session()

#Declare model parameters
batch_size = 10000
vocabulary_size = 200000
generations = 150000
model_learning_rate = 0.001

embedding_size = 400     # Word embedding size
doc_embedding_size = 300
concatenated_size = embedding_size + doc_embedding_size

num_sampled = int(batch_size/2) # Number of negative examples to samples
window_size = 2 # Numbers to consider to the left

#Add checkpoint to training
save_embeddings_every = 5000
print_valid_every = 100
print_loss_every = 50
checkpoint_path = "doc2vec_mbti_tweets.ckpt"
dictionary_path = "tweet_vocab.pkl"
number_of_tweets = len(texts)

#Validation words
valid_words = ["il","elle","grand","petit","homme","femme","roi","reine","malade","sex","voir","vue","fille","garcon","pense","musique","facebook","réseaux","regarder","tele","ecole","maman","mére","classe","article","directeur","direction","2015","année","langue","poid","tirer","croire","savoir","force","sms"]

# Define Embeddings:
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
doc_embeddings = tf.Variable(tf.random_uniform([number_of_tweets, doc_embedding_size], -1.0, 1.0))

#Importing the dictionaries and pre-trained Embeddings
print('Dictionary Backup')
with open("".join([save_data_folder,dictionary_path]),"rb") as f:
    word_dictionary = pickle.load(f)
word_dictionary_rev = dict(zip(word_dictionary.values(), word_dictionary.keys())) # Import dictionary

text_data = text_to_numbers(texts, word_dictionary)
print('Printing first 2 tweet numbers from dictionnary {} '.format(text_data[:2]))

#Get Validation word Keys declared above
valid_examples = [word_dictionary[x] for x in valid_words] 
print('Creating Model')

# NCE loss parameters
nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, concatenated_size], stddev=1.0 / np.sqrt(concatenated_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

# Create data/target placeholders
x_inputs = tf.placeholder(tf.int32, shape=[None, window_size + 1]) # plus 1 for doc index
y_target = tf.placeholder(tf.int32, shape=[None, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

# Lookup the word embedding
# Add together element embeddings in window:
embed = tf.zeros([batch_size, embedding_size])
for element in range(window_size):
    embed += tf.nn.embedding_lookup(embeddings, x_inputs[:, element])

doc_indices = tf.slice(x_inputs, [0,window_size],[batch_size,1])
doc_embed = tf.nn.embedding_lookup(doc_embeddings,doc_indices)

# concatenate embeddings
final_embed = tf.concat([embed, tf.squeeze(doc_embed)],1)

# Get loss from prediction
loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, y_target, final_embed, num_sampled, vocabulary_size))

# Create optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=model_learning_rate)
train_step = optimizer.minimize(loss)

# Cosine similarity between words
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

# Create model saving operation
saver = tf.train.Saver({"embeddings": embeddings, "doc_embeddings": doc_embeddings})

#Add variable initializer.
init = tf.global_variables_initializer()
sess.run(init)

# Run the skip gram model.
print('Starting Training')
loss_vec = []
loss_x_vec = []
for i in range(generations):
    batch_inputs, batch_labels = generate_batch_data(text_data, batch_size, window_size)
    feed_dict = {x_inputs : batch_inputs, y_target : batch_labels}

    # Run the train step
    sess.run(train_step, feed_dict=feed_dict)

    # Return the loss
    if (i+1) % print_loss_every == 0:
        loss_val = sess.run(loss, feed_dict=feed_dict)
        loss_vec.append(loss_val)
        loss_x_vec.append(i+1)
        print('Loss at step {} : {}'.format(i+1, loss_val))
      
    # Validation: Print some random words and top 5 related words
    if (i+1) % print_valid_every == 0:
        sim = sess.run(similarity, feed_dict=feed_dict)
        for j in range(len(valid_words)):
            valid_word = word_dictionary_rev[valid_examples[j]]
            top_k = 5 # number of nearest neighbors
            nearest = (-sim[j, :]).argsort()[1:top_k+1]
            log_str = "Nearest to {}:".format(valid_word)
            for k in range(top_k):
                close_word = word_dictionary_rev[nearest[k]]
                log_str = '{} {},'.format(log_str, close_word)
            print(log_str)
            
    # Save dictionary + embeddings
    if (i+1) % save_embeddings_every == 0:
        # Save vocabulary dictionary
        with open(os.path.join(save_data_folder,'tweet_vocab.pkl'), 'wb') as f:
            pickle.dump(word_dictionary, f)
        
        # Save embeddings
        model_checkpoint_path = os.path.join(os.getcwd(),save_data_folder,'doc2vec_mbti_tweets.ckpt')
        save_path = saver.save(sess, model_checkpoint_path)
        print('Model saved in file: {}'.format(save_path))


