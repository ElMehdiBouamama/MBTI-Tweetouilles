import sys
sys.path.append('C:/Users/BOUÂMAMAElMehdi/documents/visual studio 2017/Projects/PythonBasics/PythonBasics/')
import numpy as np
from DatabaseManager import *
import matplotlib.pyplot as plt
from multiprocessing import Pool
import collections
import os

ProjectFolder = "C:/Users/BOUÂMAMAElMehdi/Documents/Visual Studio 2017/Projects/PythonBasics/PythonBasics/"
DataFolder = ProjectFolder + "ExtractedTweets/"
datas = ReadJsonFile(ProjectFolder + "TwiSty-FR.json")
userIds = np.loadtxt(ProjectFolder + 'ValidUserIds.txt' ,dtype=np.str)

vocabulary = []

def ReadFiles(fileName):
    TempTweets = [];
    with open(DataFolder + fileName + ".txt", "r", encoding="UTF-8") as f:
        Tweets = f.readlines()
        for Tweet in Tweets:
            TempTweets = TempTweets + Tweets.split(" ")
    return TempTweets

with Pool(150) as p:
    spareVocabulary = p.map(ReadFiles,userIds)
for UserVocab in spareVocabulary:
    vocabulary = vocabulary + UserVocab
len(vocabulary)



vocabulary_Size = len(vocabulary)/1405

def build_dataset(words, n_words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

data,count,dictionnary,reversed_dictionnary = build_dataset(vocabulary,vocabulary_Size)

#Reduce memory
del vocabulary
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0

#TODO : Modify this function to only scrap and verify windows for each user
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]
            if data_index == len(data):
                buffer[:] = data[:span]
                data_index = span
            else:
                buffer.append(data[data_index])
                data_index += 1
    #Backtrack to avoid missing a word at the end of the batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
#Display some batches and their reverse dictionnaries to see them
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]],
    '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

#Variables for training
batch_size = 128
embeding_size = 512
skip_window = 5
num_skips = 2
num_samples = 64
#Variables for Verification
valid_size = 16 
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)


'''
# Building dictionnary of types
MBTI = [GetMbtiOfUser(datas,x) for x in userIds]
MBTITypes = []
#Initialize the dictionnary with 16 keys
typesCount = collections.Counter(MBTI).most_common()
for k in typesCount:
    MBTITypes.append(k[0])
del typesCount
#Initialize the keys with 16 arrays
MbtiDict = dict({MBTITypes[x]:[] for x in range(len(MBTITypes))})
#Fill the keys with respective values
for i,j in enumerate(MBTI):
    MbtiDict[j].append(userIds[i])

z = [len(MbtiDict[MBTITypes[x]])/len(userIds) for x in range(16)]
plt.bar(MBTITypes,z)
plt.show()
'''
len(result)