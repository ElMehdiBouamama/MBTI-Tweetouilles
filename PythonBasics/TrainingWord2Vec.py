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
    TempTweets = []
    with open(extracted_tweet_folder + fileName + ".txt", "r", encoding="UTF-8") as f:
        Tweets = f.readlines()
        for Tweet in Tweets:
            TempTweets = TempTweets + Tweet.split(" ")
    return TempTweets

#%% cell 1
with Pool(150) as p:
    users_vocab = p.map(ReadFiles,userIds)

sess = tf.Session()

#Declare model parameters
batch_size = 1000
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
print_valid_every = 5000
print_loss_every = 100

#Validation words
valid_words = ["il","elle","grand","petit","homme","femme"]

#Creating the dictionnaries
print('Creating Dictionary')
word_dictionary = build_dictionary(users_vocab, vocabulary_size)
word_dictionary_rev = dict(zip(word_dictionary.values(), word_dictionary.keys()))
texts = split_tweets_into_sentences(users_vocab)
print('First 2 tweets:')
for x in texts[:2]:
    print(x)
text_data = text_to_numbers(texts, word_dictionary)
print('First tweet vectors:')
print(text_data[0])
print('Tweet Number:')
print(len(texts))

#Get Validation word Keys declared above
valid_examples = [word_dictionary[x] for x in valid_words] 
print('Printing valid example:')
print(valid_examples)
print('Creating Model')
# Define Embeddings:
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
doc_embeddings = tf.Variable(tf.random_uniform([len(texts), doc_embedding_size], -1.0, 1.0))

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

