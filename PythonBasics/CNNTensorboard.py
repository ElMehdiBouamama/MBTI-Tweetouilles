#%% cell 0
#Adding local path to sys paths to enable local project imports
import sys
sys.path.append('C:/Users/BOUÂMAMAElMehdi/documents/visual studio 2017/Projects/PythonBasics/PythonBasics/MBTI/')
from pprint import pprint
from DatabaseManager import *
from TwitterManager import *
from nltk.tokenize import TweetTokenizer
import collections
import tensorflow  as tf

TweetFolderPath = "M:/Datasets/TwitterMBTI/Tweets/"
#Reading data from the Json File and extracting datas
datas = ReadJsonFile("M:/Datasets/TwitterMBTI/MBTINotExtracted/twisty-2016-03/TwiSty-FR.json")
numberOfUsers = 1405
numberOfTweets = 1995865.
vocabulary_size = 50000
numberOfTweetsPerFeedBack = 500

UserIds = GetUserIds(datas)
Genders =  [GetGenderOfUser(datas,UserIds[x]) for x in range(numberOfUsers)]
Mbtis =  [GetMbtiOfUser(datas,UserIds[x]) for x in range(numberOfUsers)]


#%% cell 1

#def extract_tweets_into_files(datas, fileName, skipToUser):
#    userIds = GetUserIds(datas)
#    tknzr = TweetTokenizer(False,True,True)
#    ConfirmedTweetCount = GetCountArrayOfConfirmedTweet(datas)
#    count = [['UNK', -1]]
#    TweetDatas = []
#    i = 0
#    j = 0
#    for userId in userIds:
#        if(j<=skipToUser):
#            i = i + ConfirmedTweetCount[j]
#            j = j + 1
#            continue
#        confirmedTweetIds = GetConfirmedTweetIdsOfUser(datas,userId)
#        print("Opening new file")
#        file = open(TweetFolderPath + fileName + str(j) + ".txt","w+", encoding="utf-8")
#        for confirmedTweetId in confirmedTweetIds:
#            SingleTweet = GetPostFromTwitter(userId,confirmedTweetId)
#            tokenizedTweet= tknzr.tokenize(SingleTweet)
#            TweetDatas = TweetDatas + tokenizedTweet
#            stringToSave = " ".join(str(x) for x in tokenizedTweet)
#            file.write(stringToSave)
#            i=i+1
#            print("Tweet %d done" % i)
#            if (i%numberOfTweetsPerFeedBack == 0):
#                count = collections.Counter(TweetDatas).most_common(vocabulary_size-1)
#                print("%s" % str((i*100.)/numberOfTweets) + " % done processing and " + "%s tweets left to extract" % str(numberOfTweets-i))
#                if(len(count) >= 200):
#                    print(count[:200])
#                del TweetDatas
#                TweetDatas = []
#        file.close()
#        j = j + 1

#extract_tweets_into_files(datas,"User")
##%% cell 0

#def build_dataset(datas):
#    userIds = GetUserIds(datas)
#    tknzr = TweetTokenizer(False,True,True)
#    TweetDatas = []
#    count = [['UNK', -1]]
#    i = 0
#    j = 0
#    file = open(TweetFolderPath + "Tweet" + str(j) + ".txt","w+", encoding="utf-8")
#    print("Done opening File")
#    for userId in userIds:
#        confirmedTweetIds = GetConfirmedTweetIdsOfUser(datas,userId)
#        for confirmedTweetId in confirmedTweetIds:
#            SingleTweet = GetPostFromTwitter(userId,confirmedTweetId)
#            tokenizedTweet= tknzr.tokenize(SingleTweet)
#            TweetDatas = TweetDatas + tokenizedTweet
#            stringToSave = " ".join(str(x) for x in tokenizedTweet)
#            file.write(stringToSave)
#            i=i+1
#            print("Tweet %d done" % i)
#            if (i%numberOfTweetsPerFile == 0):
#                file.close()
#                print("Opening new file")
#                j = j + 1
#                file = open(TweetFolderPath + "Tweet" + str(j) + ".txt","w+", encoding="utf-8")
#                count = collections.Counter(TweetDatas).most_common(vocabulary_size-1)
#                print("%d done processing %d left tweets" % ((i*100.)/numberOfTweets , numberOfTweets-i))
#                if(len(count) >= 200):
#                    print(count[:500])

#    print("Done downloading tweets and moving them to memory")
#    dictionnary = dict()
#    for _,word in count:
#        dictionary[word] = len(dictionary)
#    data = list()
#    unk_count = 0
#    for word in TweetDatas:
#        if word in dictionnary:
#            index = dictionary[word]
#        else:
#            index = 0
#            unk_count = unk_count + 1
#        data.apprend(index)
#    count[0][1] = unk_count
#    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
#    print("Done building dictionnary and reverse dictionnary")
#    return data, count, dictionary, reverse_dictionary

#data,count,dictionnary,reverse_dictionnary = build_dataset(datas)


##%% cell 1
#for userId in userIds:
#    TweetIds = GetConfirmedTweetIdsOfUser(datas,userId)
#    file = open("Tweets." + userId + "txt","w+", encoding="utf-8")
#    UserInfo = "Gender : " + GetGenderOfUser(datas,userId) + " Type :" + GetMbtiOfUser(datas,userId) + "\n"
#    file.writelines(UserInfo)
#    for TweetId in TweetIds:
#        UserTweet = "    " +  GetPostFromTwitter(userId,TweetId) + "\n"
#        pprint(UserTweet)
#    file.close()
#    del file



