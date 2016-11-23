
import pandas as pd

#this result is from google big query
#https://github.com/HackerNews/API for fields definition
#!!! I need more data...
df = pd.read_csv('/Users/jing/Desktop/GA/Project_GA/results-20161120-154144.csv')

##################### CLEANING DATA #####################

#parent, ranking are all null. drop those columns.
df.drop(['parent', 'ranking'], axis=1,inplace = True)

#time is in unix, change it to mm/dd/yy hh:mm:ss 
df=df.set_index(pd.to_datetime(df['time'], unit='s'))

#deleted and dead are both boolean. those null = False. transform it to 1 and 0
df['deleted'].fillna(value=False, inplace = True)
df['dead'].fillna(value=False, inplace  = True)
df[['dead','deleted']]=df[['dead','deleted']].astype(int)

#take care of null
df['descendants'].fillna(value =0, inplace =True)
#!!! fill df['type'] with existing values base on frequency
df['type'].fillna(value = 'job', inplace = True)
df['score'].fillna(value = df['score'].mean(), inplace = True)

#get_dummies for type: job, poll, story
type_dum =pd.get_dummies(df['type'], drop_first =True)
df=pd.concat([df,type_dum], axis=1)
df = df.sort_index()

#map each type to numeric
df['type_numeric']= df['type'].map({'job' :0,'story':1, 'poll':2})

#add len of text to part of df
df['text_length'] = df['text'].str.len()
#creating column for hour when story is posted
df['post_hour'] = pd.DatetimeIndex(df.index).hour
df['post_hour'].fillna(value = df['post_hour'].mean(), inplace = True)
df['post_day'] = pd.DatetimeIndex(df.index).dayofweek #0 for monday
df['post_day'].fillna(value = df['post_day'].mean(), inplace = True)

#check for null
#print df.info()
#print df.head()

##################### VISUALISATION #####################

import matplotlib.pyplot as plt

#which Q period has the most hackernews
bar_chart=df['id'].groupby(pd.TimeGrouper("Q")).count().plot(kind='bar', title='No. of Hackernews per month')
bar_chart.x_label = 'Date'
bar_chart.y_label = 'Quantity'
fig = bar_chart.get_figure()
fig.savefig('y.png')


#groupby scores and count stories. check what is the most common score for a story
score_chart = df['id'].groupby(df['score']).count().plot(kind ='bar',title='Score for story')
score_chart.set_xlabel('Score')
score_chart.set_ylabel('Number of stories')
fig2 = score_chart.get_figure()
fig2.savefig('Score_stories.png')

#which authors consistently get on the front page
average_score = df['score'].groupby(df['by']).mean()
print "\n"
print 'Top 5 Authors base on mean score:' , average_score.sort_values(ascending = False).head(5) #top 5 author with highest score
print "\n"

'''
#!!! Nan for score and type...
#scatter plot score vs type
score_type = df.plot(kind = 'scatter', x = df['type_numeric'], y = df['score'], alpha = 0.2)
score_type.set_xlabel('type')
score_type.set_ylabel('score')
fig3 = score_type.get_figure()
fig3.savefig('Score_type.png')
'''


##################### TEXT ANALYSIS USING NAIVE BAYES #####################

#ngram text analysis on 68 text. 
#print df.head()
from sklearn.feature_extraction.text import CountVectorizer

CV = CountVectorizer(ngram_range=(1,2))
df_text = df[['score', 'text']].copy()
df_text.dropna(axis=0, inplace = True)
X = CV.fit_transform(df_text['text'])
Y = df_text['score']

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X,Y, random_state =3)

def accuracy_report(_clf):
    training_accuracy = _clf.score(xtrain, ytrain)
    test_accuracy = _clf.score(xtest, ytest)
    print "Accuracy on test data: %0.2f%%" % (100 * test_accuracy)
    print "Accuracy on training data: %0.2f%%" % (100 * training_accuracy)

print '===Naive Bayes: Feature = text; Target = score.===\n'


from sklearn.naive_bayes import MultinomialNB
print "MultinomialNB:"
clf_m = MultinomialNB().fit(xtrain, ytrain)
accuracy_report(clf_m)

print "\n"

from sklearn.naive_bayes import BernoulliNB
print "BernoulliNB:"
clf_b = BernoulliNB().fit(xtrain, ytrain)
accuracy_report(clf_b)

print "\n"

from sklearn.linear_model import LogisticRegression
print "Logistic Regression:"
clf_lr = LogisticRegression().fit(xtrain, ytrain)
accuracy_report(clf_lr)


#find the most common words among highest score.

print "============"
print df[df['text'].notnull()].shape

##################### KNN/Kfold ##################### 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import numpy as np
print df.info()

X = df[['deleted', 'dead', 'descendants','poll','story', 'post_hour','post_day']]
y = np.asarray(df.score, dtype ='|S6')


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X,y)
print knn.predict([[1,0,0,0,1,10,4]])

from sklearn.model_selection import Kfold

