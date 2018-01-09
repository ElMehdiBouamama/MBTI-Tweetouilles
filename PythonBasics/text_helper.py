#%% cell 0
import string
import collections
import numpy as np
from multiprocessing import Pool
from multiprocessing import Manager


manager = Manager()
cleared_texts = manager.list([])

#TODO: Parallelize this task
def remove_carriage_return(word):
    if(word != "\n"):
        cleared_texts.append(word)

#Normalize text
def normalize_text(texts):
    #Lower Case
    texts = [x.lower() for x in texts]
    #Remove Carriage Return
    with Pool(25) as p:
        lol = p.map(remove_carriage_return,texts)
    print(cleared_texts[:50])
    #Counting Punction and Emoji
    punction_count = 0
    emoji_count = 0
    for x in texts:
        for c in x:
            if(c in string.punctuation):
                punction_count = punction_count + 1
            elif(c not in string.digits and c not in string.ascii_letters):
                emoji_count = emoji_count + 1
    del texts
    return cleared_texts, punction_count, emoji_count

# Build dictionary of words
def build_dictionary(users_Tweets, vocabulary_size):
    # Turn sentences (list of strings) into lists of words
    words = [x for user_Tweets in users_Tweets for x in user_Tweets]
    # Normalize Words
    words,_,_ = normalize_text(words) 
    # Initialize list of [word, word_count] for each word, starting with unknown
    count = [['RARE', -1]]
    
    # Now add most frequent words, limited to the N-most frequent (N=vocabulary size)
    count.extend(collections.Counter(words).most_common(vocabulary_size-1))
    
    # Now create the dictionary
    word_dict = {}
    # For each word, that we want in the dictionary, add it, then make it
    # the value of the prior dictionary length
    for word, word_count in count:
        word_dict[word] = len(word_dict)
    
    return(word_dict)

# Turn tweets per user list to sentences list
def split_tweets_into_sentences(user_Tweets):
    sentences = []
    for tweets in user_Tweets:
        temp = []
        for word in tweets:
            if(word != "\n"):
                temp.append(word)
            else:
                sentences.append(temp)
                temp = []
    return(sentences)


# Turn text data into lists of integers from dictionary
def text_to_numbers(users_Tweets, word_dict):
    # Initialize the returned data
    data = []
    for sentences in users_Tweets:
        sentence_data = []
        # For each word, either use selected index or rare word index
        for word in sentences:
            if word in word_dict:
                word_ix = word_dict[word]
            else:
                word_ix = 0
            sentence_data.append(word_ix)
        data.append(sentence_data)
    return(data)


# Generate data randomly (N words behind, target, N words ahead)
def generate_batch_data(sentences, batch_size, window_size):
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
    print(batch_data[:2])
    batch_data = batch_data[:batch_size]
    label_data = label_data[:batch_size]
    # Convert to numpy array
    batch_data = np.array(batch_data)
    label_data = np.transpose(np.array([label_data]))
    
    return(batch_data, label_data)
