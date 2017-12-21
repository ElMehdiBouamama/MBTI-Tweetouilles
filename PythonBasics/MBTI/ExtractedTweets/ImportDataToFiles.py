import sys
sys.path.append('C:/Users/BOUÂMAMAElMehdi/documents/visual studio 2017/Projects/PythonBasics/PythonBasics/MBTI/')
from pprint import pprint
from DatabaseManager import *
from TwitterManager import *
from nltk.tokenize import TweetTokenizer
import collections
import tensorflow  as tf
from multiprocessing import Pool

TweetFolderPath = "M:/Datasets/TwitterMBTI/TweetsParal/"
datas = ReadJsonFile("M:/Datasets/TwitterMBTI/MBTINotExtracted/twisty-2016-03/TwiSty-FR.json")
