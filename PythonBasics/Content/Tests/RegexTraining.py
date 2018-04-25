#%% cell 0
import re
from urllib.request import Request, urlopen

url = "https://samsung.jumia.ma/"

def ExtractDataFromURL(url):
    req = Request(url)
    req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36')
    req.add_header('Referer', 'hchouma')
    req.add_header('Cookie', 'JSC=3bP3wuFX6QdxggjsaWyoDQku%2B9HbznIklJ%2Bn4nmMFdoCqFJrHRYqPJZni2RHuEiNo9u3j8K%2BPZSWKx6ICfoo3SjC0fJcpohhCPMJP4UOWSLl8VLrKjbim2vogjCj8G2ZNT7gnL5th8xqsRkc0oRvQlJvaAnNnzuY9G59ZGn%2FazqThFuvlPzP%2B8NZrLilkRuJGNjO2bpk16g6UwwSDlwY8Q%3D%3D; PHPSESSID_52c23ed37b29d89661d4a206a2e91ad0=tk86tjthh95dorab3lcogsfah0; __asc=c86b59b916290a1435176b71a6e; __auc=2442f1b2161ed539ca241e180a3; __cfduid=d6233cdff788a27d3ef6da265db7b04e31520105122; browserDetection=eyJ0eXBlIjoiYnJvd3NlciIsIm5hbWUiOiJDaHJvbWUiLCJjc3NDbGFzcyI6ImNocm9tZSIsInZlcnNpb24iOiI2NSJ9; cto_lwid=75691c1f-8296-416f-9117-005696689b99; customerType=new; mi_gclid=CjwKCAiAlfnUBRBQEiwAWpPA6X_6wq-UZ7Ki5C_DLr50pAzPXJsMVAPpihTCFhQ9vsgeiDHkmeTz2hoC9YIQAvD_BwE; userId=881406; _ga=GA1.2.227326580.1520105135; _gac_UA-32132208-1=1.1520354325.CjwKCAiAlfnUBRBQEiwAWpPA6X_6wq-UZ7Ki5C_DLr50pAzPXJsMVAPpihTCFhQ9vsgeiDHkmeTz2hoC9YIQAvD_BwE; _gac_UA-78960438-8=1.1520354325.CjwKCAiAlfnUBRBQEiwAWpPA6X_6wq-UZ7Ki5C_DLr50pAzPXJsMVAPpihTCFhQ9vsgeiDHkmeTz2hoC9YIQAvD_BwE; _gid=GA1.2.1309902688.1522844910; accountType=Customer; path=/; uid=a1cf490f-2913-4c4a-b700-1974449f2a4c')
    with urlopen(req) as f:
        data = str(f.read())
    priceAndName = re.findall('<div.*?class="sku -gallery.*?<span class="name".*?>(.*?)</span>.*?<span.*?data-price="(\d+)".*?</span>.*?</div>',data)
    return priceAndName

AllPrices = [ExtractDataFromURL("https://www.jumia.ma/vetements-femmes-mode/?page=" + str(x+1)) for x in range(24)]

