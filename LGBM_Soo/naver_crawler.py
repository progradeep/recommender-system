# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import urllib.request
from urllib.parse import quote
import json
import re
import requests, time
import pandas as pd

#네이버 검색 Open API 사용 요청시 얻게되는 정보를 입력합니다
naver_client_id = "wXxPslZJGJrplu5FPfKk"
naver_client_secret = "4kkKrmJzRF"

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

def searchByTitle(title):
    myurl = 'https://openapi.naver.com/v1/search/movie.json?display=100&query=' + quote(title)
    request = urllib.request.Request(myurl)
    request.add_header("X-Naver-Client-Id",naver_client_id)
    request.add_header("X-Naver-Client-Secret",naver_client_secret)
    response = urllib.request.urlopen(request)
    rescode = response.getcode()
    if(rescode==200):
        response_body = response.read()
        d = json.loads(response_body.decode('utf-8'))
        if (len(d['items']) > 0):
            return d['items']
        else:
            return None

    else:
        print("Error Code:" + rescode)

def findItemByInput(items, meta,year,title):
    total_len = len(items)
    compare = []
    for index, item in enumerate(items):
        navertitle = cleanhtml(item['title'])
        naversubtitle = cleanhtml(item['subtitle'])
        naverpubdate = cleanhtml(item['pubDate'])
        naveractor = cleanhtml(item['actor'])
        naverlink = cleanhtml(item['link'])
        naveruserScore = cleanhtml(item['userRating'])
        naverDirector = cleanhtml(item['director'])

        naveractor = ",".join(naveractor.split("|")[:-1])
        naverDirector = ",".join(naverDirector.split("|")[:-1])






        navertitle1 = navertitle.replace(" ","")
        navertitle1 = navertitle1.replace("-", ",")
        navertitle1 = navertitle1.replace(":", ",")

        #기자 평론가 평점을 얻어 옵니다
        # spScore = getSpecialScore(naverlink)

        #네이버가 다루는 영화 고유 ID를 얻어 옵니다다
        naverid = re.split("code=", naverlink)[1]

        if str(title).replace(" ","") == str(navertitle).replace(" ",""):
            if (naverpubdate != "" and int(naverpubdate) == year) or (meta in naverDirector) or index+1==total_len:
                print(navertitle, meta, naverDirector, naverpubdate, year)
                compare.append([naverid, navertitle, naversubtitle, naverpubdate, naveractor, naveruserScore, naverDirector])
                break

        # 영화의 타이틀 이미지를 표시합니다
        # if (item['image'] != None and "http" in item['image']):
        #    response = requests.get(item['image'])
        #    img = Image.open(BytesIO(response.content))
        #    img.show()
    if len(compare) > 0: return compare[-1]
    return compare

def getInfoFromNaver(searchTitle,meta,year,title):
    items = searchByTitle(searchTitle)
    if (items != None):
        return findItemByInput(items,meta,year,title)
    else:
        return []

def get_soup(url):
    source_code = requests.get(url)
    plain_text = source_code.text
    soup = BeautifulSoup(plain_text, 'lxml')
    return soup

#기자 평론가 평점을 얻어 옵니다
# def getSpecialScore(URL):
#     soup = get_soup(URL)
#     scorearea = soup.find_all('div', "spc_score_area")
#     newsoup = BeautifulSoup(str(scorearea), 'lxml')
#     score = newsoup.find_all('em')
#     if (score and len(score) > 5):
#         scoreis = score[1].text + score[2].text + score[3].text + score[4].text
#         return float(scoreis)
#     else:
#         return 0.0

def clean(text):
    y = re.compile("\([1-9]*\)")
    year = y.findall(str(text))
    p = re.compile('\[.*?\]|\(.*?\)')
    cleantext = p.sub("",str(text))
    if len(year) != 0:
        return cleantext, int(year[0][1:-1])
    else: return cleantext, 0

movie_data = pd.read_excel("C:\\Users\msi\Desktop\Soohyun\CHALLENGERS\TBCC\Final_DATA\\tbcc_combined.xlsx")
output = []
for movie in movie_data.values[:]:
    mid = int(movie[0])
    print(mid)
    title = movie[1]
    title, year = clean(title)
    meta = str(movie[6]).split(',')[0]
    # print(meta)
    ret = getInfoFromNaver(u"%s"%title,meta,year,title)
    if len(ret) > 0:
        ret = [mid] + ret
        output.append(ret)
    time.sleep(0.1)

df = pd.DataFrame(output)
print(df)
df.to_excel("naver_crawled.xlsx",index_label=False)
# getInfoFromNaver(u"007 제1탄-살인번호")