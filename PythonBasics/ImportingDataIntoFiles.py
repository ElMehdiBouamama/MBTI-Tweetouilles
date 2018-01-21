#%% cell 0
from multiprocessing import Pool
import sys
import os
sys.path.append(os.getcwd())
from DatabaseManager import *
from TwitterManager import *
from functools import partial

TweetFolderPath =  os.getcwd() + "/OtherExtractedTweets/"
datas = ReadJsonFile("./TwiSty-FR.json")
userIds = GetUserIds(datas)
os.chdir(TweetFolderPath)
# We need to build a dictionnary of users / tweets first

for x in userIds[240:]:
    print("Processing user : " + x)
    with Pool(2) as p:
        UserTweets = GetOtherTweetIdsOfUser(datas,x)
        checker = None
        if(UserTweets):
            checker = GetPostFromTwitter(UserTweets[0],x)
        print(checker)
        if(checker):
            GetPostPartial = partial(GetPostFromTwitter, userId=x)
            result =  p.map(GetPostPartial,UserTweets)
        else:
            result = ""
    with open(x + ".txt","w+", encoding="utf-8") as file:
        file.writelines(result)

