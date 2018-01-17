#%% cell 0
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from pandas.plotting import scatter_matrix
import pandas

# Paths
folder_path = "C:\\Users\\BOUÂMAMAElMehdi\\documents\\visual studio 2017\\Projects\\PythonBasics\\PythonBasics\\SaveFolder\\"
checkpoint_path = "doc2vec_mbti_tweets.ckpt"
dictionnary_path = "tweet_vocab.pkl"

# Variable sizes
vocabulary_size = 200000
embedding_size = 400 
doc_embedding_size = 300 
number_of_tweets = 1586307
# Variables
embeddings = tf.Variable(tf.zeros([vocabulary_size, embedding_size]))
doc_embeddings = tf.Variable(tf.zeros([number_of_tweets, doc_embedding_size]))

#%% cell 1
# Restore data from files
with open("".join([folder_path,dictionnary_path]),"rb") as f:
    word_dictionnary = pickle.load(f)
word_dictionary_rev = dict(zip(word_dictionnary.values(), word_dictionnary.keys()))
saver = tf.train.Saver({"embeddings": embeddings, "doc_embeddings": doc_embeddings})
sess = tf.Session()
saver.restore(sess, "".join([folder_path,checkpoint_path]))


#%%cell 2
#Setting for K most common word values
k = 1000
k_most_common_word_vectors = [tf.nn.embedding_lookup(embeddings,x) for x in range(k)]
display_values = sess.run(k_most_common_word_vectors)
display_labels = [word_dictionary_rev[x] for x in range(k)]

#function to display data
def DisplayData(components,labels):
    x = components[:,0]
    y = components[:,1]
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    for i,txt in enumerate(labels):
        ax.annotate(txt, (x[i],y[i]))
    plt.show()

# Runing PCA
#%% cell 3
pca = PCA(n_components=2)
data_points = pca.fit_transform(display_values)
covMatrix = pandas.DataFrame(pca.get_covariance())
#scatter_matrix(covMatrix[:1])
#plt.show()
#DisplayData(data_points,display_labels)

#%% cell 4
data_points = TSNE(n_components=2,perplexity=500).fit_transform(display_values)
DisplayData(data_points,display_labels)



