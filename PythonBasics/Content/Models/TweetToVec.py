import tensorflow as tf
import numpy as np
import Content.Managers.DataManager as DataManager
import pickle
from Content.Helpers.text_helper import *
from tensorflow.contrib.tensorboard.plugins import projector

class TweetToVec(object):
    def __init__(self):
        # Imports data from files
        self.dataman = DataManager.DataManager()
        self.confman = self.dataman.configman
        pass

    def Fit(self, texts):
        ''' Return word Embeddings and Documents Embeddings'''
        sess = tf.Session()
        # All this section should be completed with adequate values from confman
        batch_size = self.confman.DTVbatch_size
        vocabulary_size = self.confman.vocabulary_size
        generations = self.confman.DTVnum_epoch
        model_learning_rate = self.confman.DTVlearning_rate
        # Embeddings properties
        embedding_size = self.confman.embedding_size
        doc_embedding_size = self.confman.doc_embedding_size
        concatenated_size = embedding_size + doc_embedding_size
        num_sampled = int(batch_size/2)
        window_size = self.confman.window_size
        # Save properties
        save_embeddings_every = self.confman.DTVsave_embeddings_every
        print_valid_every = self.confman.DTVprint_valid_every
        print_loss_every = self.confman.DTVprint_loss_every
        number_of_tweets = len(texts)
        #Validation words
        valid_words = ["il","elle","grand","petit","homme","femme","roi","reine","malade"]
        # Define Embeddings:
        embeddings = tf.Variable(tf.random_uniform([self.confman.vocabulary_size,self.confman.embedding_size], -1.0, 1.0), name="word_embeddings")
        saver = tf.train.Saver({"embeddings": embeddings})
        saver.restore(sess, self.confman.checkpoint_path)
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embeddings.name

        doc_embeddings = tf.Variable(tf.random_uniform([number_of_tweets, doc_embedding_size], -1.0, 1.0), name="doc_embeddings")
        #Importing dictionnaries
        self.dataman.restore_dictionaries()
        text_data = text_to_numbers(texts, self.dataman.word_dictionary)
        print("Printing first 2 tweet numbers from dictionnary {}".format(text_data[:2]))
        # Get Validation word keys declared above
        valid_examples = [self.dataman.word_dictionary[x] for x in valid_words]
        print("Creating Model")
        # NCE loss parameters
        nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, concatenated_size], stddev=1.0 / np.sqrt(concatenated_size)), name="nce_weights")
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]), name="nce_biases")
        # Create data/target placeholders
        x_inputs = tf.placeholder(tf.int32, shape=[None, window_size + 1])
        y_target = tf.placeholder(tf.int32, shape=[None, 1])

        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
        # Lookup the word embedding
        # Add together element embeddings in window:
        embed = tf.zeros([batch_size, embedding_size])
        for element in range(window_size):
            embed += tf.nn.embedding_lookup(embeddings, x_inputs[:, element])
        doc_indices = tf.slice(x_inputs, [0,window_size], [batch_size, 1])
        doc_embed = tf.nn.embedding_lookup(doc_embeddings, doc_indices)
        # concatenate embeddings
        final_embed = tf.concat([embed, tf.squeeze(doc_embed)], 1)
        # get loss from prediction
        loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, y_target, final_embed, num_sampled, vocabulary_size))
        # Create optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=model_learning_rate)
        train_step = optimizer.minimize(loss)
        # Cosine similarity between words
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
        # Create model saving operations
        del saver
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter("/mnt/m/hihi", sess.graph)
        projector.visualize_embeddings(writer, config)
        # Add variable initializer
        tf.summary.scalar('loss', loss)
        tf.summary.histogram("nce_weights", nce_weights)
        tf.summary.histogram("nce_biases", nce_biases)
        summaries = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        sess.run(init)
        # Run the skip gram model
        print("Starting Training")
        loss_vec = []
        loss_x_vec = []
        #TODO: generations instead of 30
        for i in range(30):
            batch_inputs, batch_labels = generate_batch_data(text_data, batch_size, window_size)
            feed_dict = {x_inputs : batch_inputs, y_target : batch_labels}
            # Run the train step
            sess.run(train_step, feed_dict=feed_dict)
            summ = sess.run(summaries, feed_dict=feed_dict)
            writer.add_summary(summ, i)
            saver.save(sess, "/mnt/m/hihi/model.ckpt", global_step=i)
            # print loss
            if (i+1) % print_loss_every == 0:
                loss_val = sess.run(loss, feed_dict=feed_dict)
                loss_vec.append(loss_val)
                loss_x_vec.append(i+1)
                print('Loss at step {} : {}'.format(i+1, loss_val))
            #Validation: Print some random words and top 5 related words
            if (i+1)% print_valid_every == 0:
                sim = sess.run(similarity, feed_dict=feed_dict)
                for j in range(len(valid_words)):
                    valid_word = self.dataman.word_dictionary_rev[valid_examples[j]]
                    top_k = 5 #number of nearest neighbours
                    nearest = (-sim[j,:]).argsort()[1:top_k+1]
                    log_str = "Nearest to {}:".format(valid_word)
                    for k in range(top_k):
                        close_word = self.dataman.word_dictionary_rev[nearest[k]]
                        log_str = '{} {},'.format(log_str, close_word)
                    print(log_str)
            if (i+1) % save_embeddings_every == 0:
                #TODO: change saver path from manual to automatic using a new function from dataManager and keep track of trainings
                save_path = saver.save(sess, "/mnt/m/hihi/")
                print("Model saved in file: {}".format(save_path))
        projector.visualize_embeddings(writer, config)
        pass

    def GenerateBatch(self, sentences, batch_size, windows_size):
        ''' Return array of couples in the form of [ (tweetVectors, target_class), (., .), ..., (., .) ] '''
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
            label_indices = [ix if ix<window_size else window_size for ix, x in enumerate(windo_sequences)]

            # Pull out center word of interest for each window and create tuple for each window
            # For dc2vec we keep LHS window only to predict target word
            batch_and_labels = [(rand_sentence[i:i+window_size], rand_sentence[i+window_size]) for i in range(0, len(rand_sentence)-window_size)]
            if(len(batch_and_labels) < 2 ):
                continue
            batch, labels = [list(x) for x in zip(*batch_and_labels)]
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
        # Return batches and labels
        return(batch_data, label_data)

    def GetFormatedData(self):
        self.dataman.get_user_tweets(self.dataman.userIds)
        return self.dataman.all_tweets()

    def GetVectors(self, documents):
        ''' Return trained tweet vectors ready for logistic regression '''
        pass
