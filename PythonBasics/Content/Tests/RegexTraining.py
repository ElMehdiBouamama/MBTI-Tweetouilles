#%% cell 0
import re
from urllib.request import Request, urlopen

url = "https://samsung.jumia.ma/"

def ExtractDataFromURL(url):
    req = Request(url)
    req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36')
    req.add_header('Referer', 'Layn3el taboune mok a weld l9e7ba we deblokilina zeb baraka men tchawchir')
    req.add_header('Cookie', '__cfduid=d28d3254feda7fa1fd53b8382a9915ec31516200472; browserDetection=eyJ0eXBlIjoiYnJvd3NlciIsIm5hbWUiOiJDaHJvbWUiLCJjc3NDbGFzcyI6ImNocm9tZSIsInZlcnNpb24iOiI2MyJ9; newsletter=1; _ga=GA1.2.187499517.1516200481; __auc=2ced259516104975343227cd183; cto_lwid=1922878e-2f51-4add-9169-f26fd6d035f8; _gac_UA-32132208-1=1.1516202560.CjwKCAiAhfzSBRBTEiwAN-ysWCyeAD7GtGoTIFihTUmL3lIJCLb0_zfspQ_ldHTqtvAuJc9ZFdX8ERoC478QAvD_BwE; _gac_UA-78960438-8=1.1516202560.CjwKCAiAhfzSBRBTEiwAN-ysWCyeAD7GtGoTIFihTUmL3lIJCLb0_zfspQ_ldHTqtvAuJc9ZFdX8ERoC478QAvD_BwE; PHPSESSID_52c23ed37b29d89661d4a206a2e91ad0=uhid6i4gkd3qgv4jidmrrqrbf3; _gid=GA1.2.2075384996.1516403862; _gat_UA-32132208-1=1; _dc_gtm_UA-78960438-8=1; _dc_gtm_UA-32132208-1=1; __asc=1824121f16110b6b068528166c4')
    with urlopen(req) as f:
        data = str(f.read())
    priceAndName = re.findall('<div.*?class="sku -gallery.*?<span class="name".*?>(.*?)</span>.*?<span.*?data-price="(\d+)".*?</span>.*?</div>',data)
    return priceAndName

AllPrices = [ExtractDataFromURL("https://www.jumia.ma/vetements-femmes-mode/?page=" + str(x+1)) for x in range(24)]

