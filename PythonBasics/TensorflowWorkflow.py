#%% cell 0
import sys
sys.path.append('C:/Users/BOUÂMAMAElMehdi/documents/visual studio 2017/Projects/PythonBasics/PythonBasics/MBTI/')
from pprint import pprint
from DatabaseManager import *
from TwitterManager import *
from nltk.tokenize import TweetTokenizer
import collections
import tensorflow  as tf
import threading


class Configs():
    TweetFolderPath = "M:/Datasets/TwitterMBTI/Tweets2/"
    UncryptedData = ReadJsonFile("M:/Datasets/TwitterMBTI/MBTINotExtracted/twisty-2016-03/TwiSty-FR.json")
    UsersCount = GetNumberOfUsers(UncryptedData)
    TotalTweetsCount = GetTotalConfirmedTweetsCount(UncryptedData)
    CumTweetCount = np.cumsum(GetCountArrayOfConfirmedTweet(UncryptedData))
    UserIds = GetUserIds(UncryptedData)
    num_threads = 5

def Extract_Tweet(i):
    # i is a tensor so we need to convert the variables into tensors
    inc = tf.Variable(0,tf.int64)
    CumTweetCount = tf.convert_to_tensor(Configs.CumTweetCount, tf.int64)
    UserIndexCondition = lambda inc: tf.logical_and(tf.less_equal(inc,tf.size(CumTweetCount)), tf.less_equal(i,CumTweetCount[inc]))
    IncOp = lambda inc: tf.add(inc,1)
    UserIndex = tf.while_loop(UserIndexCondition, IncOp, [inc])
    tf.assign(inc,0)
    ArrayOfConfirmedTweets = tf.convert_to_tensor(GetCountArrayOfConfirmedTweet(Configs.UncryptedData), tf.int64)
    createTweedIndex = lambda: i - ArrayOfConfirmedTweets[UserIndex-1]
    TweetIndex = tf.cond(tf.not_equal(UserIndex,0), createTweedIndex, lambda: i)
    UserIds = tf.convert_to_tensor(Configs.UserIds, tf.string)
    UserId = UserIds[tf.cast(UserIndex,tf.int64)]
    with tf.Session() as sess1 :
        sess1.run(tf.global_variables_initializer())
        EUserId = sess1.run(UserId, dict_values:{inc})
        ETweetIndex = sess1.run(TweetIndex)
    TweetId = GetConfirmedTweetIdsOfUser(tf.convert_to_tensor(Configs.UncryptedData),EUserId)[ETweetIndex]
    return GetPostFromTwitter(UserId,TweetId)


# Dataset and paralezing task
batch_size = GetCountArrayOfConfirmedTweet(Configs.UncryptedData)[0]
maxQueueSize = np.max(GetCountArrayOfConfirmedTweet(Configs.UncryptedData))

dataset = tf.data.Dataset.range(Configs.TotalTweetsCount)
dataset = dataset.map(lambda x : Extract_Tweet(x))
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

q = tf.FIFOQueue(capacity = maxQueueSize,shapes = next_element.shape, dtypes = next_element.dtype)
enqueue = q.enqueue(next_element)
inputs = q.dequeue_many(batch_size)

qr = tf.train.QueueRunner(q, [enqueue] * Configs.num_threads)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    enqueue_threads = qr.create_threads(sess,coord,start=True)
    for step in range(1):
        batch_size = GetCountArrayOfConfirmedTweet(Configs.UncryptedData)[step]
        print(sess.run(inputs))
    coord.request_stop()
    coord.join(enqueue_threads)



