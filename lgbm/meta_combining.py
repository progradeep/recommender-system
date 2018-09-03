import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import urllib.request
from urllib.parse import quote
import re
import requests, time

# tbcc_meta = pd.read_excel("C:\\Users\msi\Desktop\Soohyun\CHALLENGERS\TBCC\Final_DATA\TBC_MOVIES_TITLE.xlsx",
#                           dtype={'MOVIE_ID':np.uint32,'TITLE':'category'})
#
# first_cr = pd.read_excel("C:\\Users\msi\Desktop\\naver_combined.xlsx",
#                          dtype={''})
# second_cr = pd.read_excel("./NAN_filled.xlsx")

no_sub = pd.read_excel("./NO_SUBTITLES.xlsx")

def get_soup(url):
    source_code = requests.get(url)
    plain_text = source_code.text
    soup = BeautifulSoup(plain_text, 'lxml')
    return soup

def crawl(naverid):
    review_html = get_soup("https://movie.naver.com/movie/bi/mi/point.nhn?code="+str(naverid))
    # content > div.article > div.mv_info_area > div.mv_info > dl > dd:nth-child(2) > p > span:nth-child(3)

    # movie basic info
    try:
        subtitle_yr = review_html.find("div",{'class':'mv_info'}).find("strong",{'class':'h_movie2'}).text
        if len(subtitle_yr) > 4:
            subtitle = subtitle_yr.split(",")[0].strip()
        else:
            subtitle = None
    except:
        subtitle = None

    # compare.append(
        # [naverid, navertitle, naversubtitle, naverpubdate, naveractor, naveruserScore, naverDirector, movie_length,
        #  rating_count])

    try:
        movie_step1 = review_html.find("div",{'id':"content"}).find('div',{'class':'article'}).\
        find('div',{'class':'mv_info_area'}).find('div',{'class':'mv_info'}).find('dl',{'class':'info_spec'}).\
        find('dd').find('p').find_all("span")

        if len(movie_step1) == 4:
            genre = movie_step1[0].text.strip().replace(" ","").replace('\t','').replace("\r","").\
                replace("\n","").split(",")
            genre = ",".join(genre)

            country = movie_step1[1].text.strip().replace(" ","").replace('\t','').replace("\r","").\
                replace("\n","").split(",")
            country = ",".join(country)

            length = movie_step1[2].text.strip()[:-1]
            year = movie_step1[3].text.strip().replace("\n","").split(".")[0]

        else:
            genre = None
            country = None
            length = None
            year = None

    except:
        genre = None
        country = None
        length = None
        year = None

    # director


    try:
        _director = review_html.find("div", {'id': "content"}).find('div', {'class': 'article'}).\
            find('div', {'class': 'mv_info_area'}).find('div', {'class': 'mv_info'}).\
            find('dl',{'class':'info_spec'}).find_all('dd')[1]


        director = _director.text.strip().split(",")
        director = [x.strip() for x in director]
        director = ",".join(director)

        # director = _director

    except:
        director = None

    # actors
    # content > div.article > div.mv_info_area > div.mv_info > dl > dd:nth-child(6) > p
    try:
        _actors = review_html.find("div", {'id': "content"}).find('div', {'class': 'article'}).\
            find('div', {'class': 'mv_info_area'}).find('div', {'class': 'mv_info'}).\
            find("dl",{'class':'info_spec'}).find_all('dd')[2]

        _actors = _actors.text.strip().split(",")
        actors = []

        for a in _actors:
            s = a.index("(")
            e = a.index(")")
            a = a[:s]+a[e+1:]
            actors.append(a.strip())

        actors = ",".join(actors)
        actors = actors.replace("더보기","")

    except:
        actors = None

    # netizen_point_tab_inner > span > em
    try:
        rating = float(review_html.find('div',{'id':"netizen_point_tab_inner"}).find("div",{"star_score"}).text)

        rating_count = review_html.find('div',{'id':"netizen_point_tab_inner"}).find('span',{'class':'user_count'}).find('em').text
        rating_count = int(rating_count.replace(",",""))

    except AttributeError:
        rating_count = 0
        rating = 0

    out = [naverid,subtitle,year,actors,rating,director,length,rating_count]
    print(out)
    return out

no_sub['NAVER_ID'] = no_sub['NAVER_ID'].fillna('n')

res = []
for movie in no_sub.values[:]:
    nid = movie[-1]
    movieid = movie[0]
    if nid != 'n':
        out = crawl(nid)
        res.append([movieid]+out)

# [naverid, navertitle, naversubtitle, naverpubdate, naveractor, naveruserScore, naverDirector, movie_length,
        #  rating_count])
res = pd.DataFrame(res)
res.to_excel("cral_with_id.xlsx")