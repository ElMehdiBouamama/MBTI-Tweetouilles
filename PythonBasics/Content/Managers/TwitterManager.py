#%% cell 1
from urllib.request import urlopen
import html5lib
from nltk.tokenize import TweetTokenizer

tknzr = TweetTokenizer(False,True,True)

def GetPostFromTwitter(tweetId,userId):
    stringToReturn = ""
    TweetUrlLink = "https://twitter.com/" + userId + "/status/" + tweetId
    try:
        f = urlopen(TweetUrlLink)
        document = html5lib.parse(f)
        result = BreadthFirstSearch(document).attrib['content'] + "\n"
        tokenizedTweet = tknzr.tokenize(result)
        tokenizedTweet.remove("”")
        tokenizedTweet.remove("“")
        for k in tokenizedTweet:
            if("http" not in k):
                stringToReturn = stringToReturn + k + " "
            else:
                stringToReturn = stringToReturn + "!LINK!" + " "
        stringToReturn = stringToReturn + "\n"
        return stringToReturn
    except :
        return ""
    

def BreadthFirstSearch(Element):
    if(Element != None):
        if(Verificator(Element)):
            return Element
        else:
            for x in Element:
                result = BreadthFirstSearch(x)
                if(result != None and Verificator(result)):
                    return result
    else:
        return None
       
def Verificator(x):
    try:
        return (x.tag == "{http://www.w3.org/1999/xhtml}meta" and x.attrib['property'] == "og:description")
    except :
        return False
    

