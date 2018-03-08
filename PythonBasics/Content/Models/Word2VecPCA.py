#%% cell 0
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from pandas.plotting import scatter_matrix
import pandas

# Paths

folder_path = "C:/Users/BOUÂMAMAElMehdi/documents/visual studio 2017/Projects/PythonBasics/PythonBasics/Logs/Training2/"
checkpoint_path = "doc2vec_mbti_tweets.ckpt"
dictionnary_path = "tweet_vocab.pkl"

# Variable sizes
vocabulary_size = 200000
embedding_size = 400 
doc_embedding_size = 300 
number_of_tweets = 1586823
# Variables
embeddings = tf.Variable(tf.zeros([vocabulary_size, embedding_size]))
doc_embeddings = tf.Variable(tf.zeros([number_of_tweets, doc_embedding_size]))

#%% cell 1
# Restore data from files
with open("".join([folder_path,dictionnary_path]),"rb") as f:
    word_dictionnary = pickle.load(f)
word_dictionary_rev = dict(zip(word_dictionnary.values(), word_dictionnary.keys())) # Import dictionnary
saver = tf.train.Saver({"embeddings": embeddings, "doc_embeddings": doc_embeddings}) # Import Embeddings
sess = tf.Session()
saver.restore(sess, "".join([folder_path,checkpoint_path]))


#%%cell 2
#Setting for K most common word values
k = 1000
k_most_common_word_vectors = [tf.nn.embedding_lookup(embeddings,x) for x in range(100,100+k)]
display_values = sess.run(k_most_common_word_vectors)
display_labels = [word_dictionary_rev[x] for x in range(100,100+k)]

#function to display data
def DisplayData(components,labels,explained_variance,name='Default'):
    x = components[:,0]
    y = components[:,1]
    fig, ax = plt.subplots()
    plt.title('Variance expliquée : {} pour les {} premiers elements'.format(np.sum(explained_variance),k))
    ax.scatter(x, y)
    for i,txt in enumerate(labels):
        ax.annotate(txt, (x[i],y[i]))
    plt.show()

# Runing PCA
#%% cell 3
pca = PCA(n_components=2)
data_points = pca.fit_transform(display_values)
DisplayData(data_points,display_labels,pca.explained_variance_,'PCA')

#%% cell 4
tsne = TSNE(n_components=2)
data_points = tsne.fit_transform(display_values)
DisplayData(data_points,display_labels,tsne.kl_divergence_,'TSNE')

#%% cell 5
# Picking k random words as reference for comparison
import random
k_words = 1000
word_indexes = [int(random.uniform(0,vocabulary_size)) for x in range(k_words)]
# Cosine similarity
valid_dataset = tf.constant(word_indexes, dtype=tf.int32)
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
# K-Nearest neighbours
sim = sess.run(similarity)
def KNN(top_k,word_indexes):
    log = ""
    for j in range(len(word_indexes)):
        valid_word = word_dictionary_rev[word_indexes[j]]
        nearest = (-sim[j, :]).argsort()[1:top_k+1]
        log_str = "Nearest to {}:".format(valid_word)
        for k in range(top_k):
            close_word = word_dictionary_rev[nearest[k]]
            log_str = '{} {},'.format(log_str, close_word)
        print(log_str)
        log = '{} {} \n'.format(log, log_str)
    return log   

logs = KNN(10,word_indexes)

norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = sess.run(embeddings / norm)
#%% cell 6
# Vector operations

def add(embedA, embedB):
    return np.add(embedA,embedB)

def sub(embedA,embedB):
    return np.subtract(embedA,embedB)

def getEmbeding(word):
    word_index = tf.constant(word_dictionnary[word], dtype=tf.int32)
    word_embed = tf.nn.embedding_lookup(embeddings, word_index)
    return sess.run(word_embed)


def NearestTo(embed):
    sim = np.dot(embed, np.transpose(normalized_embeddings))
    nearest = (-sim[:]).argsort()
    return [word_dictionary_rev[nearest[x]] for x in range(10000)]

manVector = getEmbeding('homme')
womenVector = getEmbeding('femme')
kingVector = getEmbeding('roi')
luiVector = getEmbeding('lui')

elleVector = add(sub(luiVector,manVector),womenVector)
queenVector = add(sub(kingVector,manVector),womenVector)

NearestTo(elleVector)
NearestTo(getEmbeding('elle'))
NearestTo(queenVector)

