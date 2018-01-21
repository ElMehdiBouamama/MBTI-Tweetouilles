#%% cell 0
from urllib.request import urlopen , Request
from urllib.parse import urlencode
import json
import re

epgCodesUrl = 'http://siptv.eu/codes/search.php'
programListPath = "M:/Mehdi.m3u"

with open(programListPath,"r") as f:
    lines = f.readlines()

FinalFile = ""
for line in lines:
        result = re.search(',(?!==*?)(.*?)$',line)
        if result != None:
            url_value = urlencode({'term': result[0][1:].replace('-',' ')})
            url = "".join([epgCodesUrl,'?', url_value])
            req = Request(url)
            req.add_header('User-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36')
            req.add_header('Cookie', '_ga=GA1.2.1323437489.1516320141; _gid=GA1.2.1445059113.1516320141; origin=valid; captcha=1')
            ParseResults= json.loads(urlopen(req).read())
            valideCandidate = ''
            for candidate in ParseResults:
                if(candidate['category'] == 'France'):
                    valideCandidate = candidate['id']
                    break
            if(valideCandidate != ''):
                line = line.replace('"ext"','"'+valideCandidate+'"')
        FinalFile = FinalFile + line
                
                
#%% cell 1

with open('M:/ModifiedMehdi.m3u','w') as f:
    f.write(FinalFile)