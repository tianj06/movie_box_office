# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 20:09:00 2015

@author: jutian
"""

#RottenTomato

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pattern import web
import pickle

#%% scrap the box office data from 'http://www.boxofficemojo.com/yearly/chart/?yr=2014&p=.htm'
tags = {'rank':'font','title':'a','studio':'a','totalGross':'b','theaterNum':'font'}
colnames =  ['rank', 'title','studio','totalGross', 'theaterNum']
boxdata = {'rank':[],'title':[],'studio':[],'totalGross':[],'theaterNum':[],'year':[]}

year_page = {'2014':8, '2013':6, '2012':6, '2011':6, '2010':5,'2009':5,'2008':6,'2007':6}
for year, pages in year_page.iteritems():
    for pageNum in range(1,pages+1):
        url = 'http://www.boxofficemojo.com/yearly/chart/?page={}&view=releasedate&view2=domestic&yr={}&p=.htm'.format(pageNum,year)
        xml = requests.get(url).text 
        dom = web.Element(xml)
        d = dom.by_tag('table')[4]
        # tags used to identify column values
        
        for tr in d.by_tag('tr'):
            if 'bgcolor' in tr.attr:        
                if tr.attr['bgcolor'] in ['#f4f4ff', '#ffffff']:
                    if len(tr.by_tag('td'))>=9:
                        for colnum, td in enumerate(tr.by_tag('td')):
                            if colnum >= len(colnames):
                                break
                            else:
                                colname = colnames[colnum]
                                boxdata[colname].append(td.by_tag(tags[colname])[0].content)
                        boxdata['year'].append(int(year))
boxdf = pd.DataFrame(boxdata)
 
# covert some columns to float                           
boxdf['totalGross'] = boxdf['totalGross'].map(lambda x: float(x.lstrip('$').replace(",","")))
boxdf['theaterNum'] = boxdf['theaterNum'].map(lambda x: x.replace("N/A","nan"))
boxdf['theaterNum'] = boxdf['theaterNum'].map(lambda x: float(x.strip().replace(",","")))
boxdf['title'] = boxdf['title'].map(lambda x: x.strip())

newCol = {'imdbID':[],'imdbRating':[]}
url = 'http://www.omdbapi.com/?'
boxdf.ix[3145,'title'] = 'Superman' 

for title in boxdf['title']:
    try:
        r = requests.get(url, params = {'t':title,'r':'json'}).json()
        newCol['imdbID'].append(r['imdbID'])
        newCol['imdbRating'].append(r['imdbRating'])
    except KeyError:
        newCol['imdbID'].append(None)
        newCol['imdbRating'].append(None)

tempNewCol = pd.DataFrame(newCol)
boxdf['imdbID'] = tempNewCol['imdbID']
boxdf['imdbRating'] = tempNewCol['imdbRating']
boxdf = boxdf.dropna()

# save the data
f = open('boxoffice.pckl','wb')
pickle.dump(boxdf, f)
f.close()

f = open('boxoffice.pckl','rb')
boxdf = pickle.load(f)
f.close()
#%% use Rotten tomatto API to get reviews
api_key = 'atgpvfr3qcnf8rj7n5syufm9'


def fetch_reviews(boxdf,row):
    imbdID = boxdf['imdbID'].iloc[row]
    if imbdID:
        imbd = imbdID[2:]
        options = {'type':'imdb','id':imbd, 'apikey': api_key}
        try:
            url = 'http://api.rottentomatoes.com/api/public/v1.0/movie_alias.json'
            id_result = requests.get(url, params=options).json()
            if 'error' in id_result:
                return None
            else: 
                try:
                    movie_id = id_result['id']        
                    url = 'http://api.rottentomatoes.com/api/public/v1.0/movies/%s/reviews.json' % movie_id
                    options = {'review_type': 'top_critic', 'page_limit': 20, 'page': 1, 'apikey': api_key}
                    data = requests.get(url, params=options).json()
                    reviews_df = pd.DataFrame(data['reviews'])
                    reviews_df = reviews_df[['critic','freshness','date','publication','quote']]
                    reviews_df['imdb'] = imbdID
                    reviews_df['title'] = id_result['title']
                    reviews_df['rtid'] = id_result['id']
                    reviews_df['publish_year'] = boxdf['year'].iloc[row]
                    reviews_df['theaterNum'] = boxdf['theaterNum'].iloc[row]
                    reviews_df['gross'] = boxdf['totalGross'].iloc[row]
                    reviews_df['critics_score'] = id_result['ratings']['critics_score']
                    reviews_df['audience_score'] = id_result['ratings']['audience_score']
                    reviews_df = reviews_df.rename(columns = {'date':'review_date','freshness':'fresh'})
                except KeyError:
                    return None
                return reviews_df
        except ValueError:
            return None
    else:
        return None

def build_table(movies, rows):
    df = list()
    for i in range(rows):
        temp = fetch_reviews(movies,i)
        if temp is not None:
            df.append(temp)
        if i%50 == 0:
            print 100*float(i)/rows
    return pd.concat(df)
    
critics = build_table(boxdf, 4000)

f = open('movie_reviews.pckl','wb')
pickle.dump(critics, f)  
f.close()
#%%
f = open('movie_reviews.pckl','rb') 
critics = pickle.load(f)
f.close()

# number of movies sucessfully retrieved
boxcritics = critics[['gross','imdb','critics_score','audience_score','title','theaterNum']]
boxcritics = boxcritics.drop_duplicates()

plt.figure()
plt.hist(np.log10(boxcritics['gross'].unique()),bins=20)
plt.xlabel('log10 (Gross box office $)')
plt.ylabel('# movies')
plt.vlines(6.5, ymin = 0, ymax=300, colors = 'r')

plt.figure()
plt.scatter(np.log10(boxcritics['gross']), boxcritics['theaterNum'])

plt.figure()
plt.scatter(boxcritics['gross'], boxcritics['critics_score'],alpha = 0.3)
plt.xscale('log')
plt.xlabel('Gross box office $')
plt.ylabel('Critic score')
plt.ylim((0,100))

plt.figure()
plt.scatter(boxcritics['gross'], boxcritics['audience_score'],alpha = 0.3)
plt.xscale('log')
plt.xlabel('Gross box office $')
plt.ylabel('Audience score')
plt.ylim((0,100))
#%%
grossthre = 3200000 # about 10**6.5 threshold distinguishing two groups of box office
boxcritics['gross_label'] = boxcritics['gross'].apply(lambda x: 1 if x>grossthre else 0)
boxcritics['log10gross'] = boxcritics['gross'].apply(np.log10)
#%% exploratory analysis
from scipy.stats.stats import pearsonr
from numpy import polyfit
from scipy.stats import ttest_ind

groupbyCluster = boxcritics.groupby('gross_label')
colors = pd.tools.plotting._get_standard_colors(2*len(groupbyCluster), color_type='random')

fig, ax = plt.subplots()
ax.set_color_cycle(colors)
names = ['low box','high box']
for name, group in groupbyCluster:
    pc = pearsonr(group['log10gross'],group['audience_score'])
    print pc
    p = polyfit(group['log10gross'],group['audience_score'],1)  
    x = np.arange(3,9,0.1)
    ax.plot(group['log10gross'], group['audience_score'], marker='o', linestyle='', 
            alpha = 0.3, label=name, color = colors[name])
    ax.plot(x, p[1]+p[0]*x, label = 'fit-'+ names[name], color = colors[name],linewidth = 2)

p = polyfit(boxcritics['log10gross'],boxcritics['audience_score'],1)  
ax.plot(x, p[1]+p[0]*x, label = 'fit-all data', color = 'k',linewidth = 2)

ax.legend(loc='lower left')

plt.xlabel('$log10(box office)')
plt.ylabel('Audience score')
plt.ylim((0,100))

pc = pearsonr(boxcritics['log10gross'],boxcritics['audience_score'])
# ttest to see whether score signficantly different between high box and low box 
a = boxcritics[boxcritics['gross_label']==1]['audience_score']
b = boxcritics[boxcritics['gross_label']==0]['audience_score']
ttest_ind(a,b)
#%% future analysis, use bag of words to predict box office
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer

alphas = [0, .1, 1, 5, 10, 50]
min_dfs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

#Find the best value for alpha and min_df, and the best classifier
best_alpha = None
best_min_df = None
max_loglike = -np.inf
def make_xy(critics, vectorizer=None):
    #Your code here   
    if vectorizer is None:
        vectorizer = CountVectorizer(min_df=0)
    vectorizer.fit(critics.quote.tolist())
    X = vectorizer.fit_transform(critics.quote.tolist())
    Y = critics.gross_label.as_matrix()
    return (X.toarray(), Y)
    
for alpha in alphas:
    for min_df in min_dfs:         
        vectorizer = CountVectorizer(min_df = min_df)       
        X, Y = make_xy(critics, vectorizer)
        print 'alpha:', alpha
        print 'min_df:', min_df
        try:
            clf = MultinomialNB(alpha=alpha)         
            loglike = cross_val_score(clf, X, Y, scoring = 'log_loss')
            print 'loglike:', loglike
            if loglike < max_loglike:
                max_loglike = loglike
                best_alpha, best_min_df = alpha, min_df
                print 'bigger'
        except MemoryError:
            print 'memory error at:', 'alpha= ', alpha, 'min_df= ', min_df



#%%
# length of title 
boxcritics['title_length'] = boxcritics['title'].apply(lambda x: len(x.split()))
plt.figure()
plt.scatter(boxcritics['title_length'], boxcritics['gross'],alpha = 0.2, label = 'data')
avg_data = boxcritics.groupby('title_length').gross.apply(np.median)
avg_data.plot(x='year', y='rtTopCriticsRating',label = 'average')
plt.yscale('log')









