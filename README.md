# Project_GA

Using data of Hackernews from Google Big Query. Query result (1000 rows) can be found in 'results-20161120-154144.csv'.

Predicting score of a new hacknew post base on features specified in [this link](https://github.com/HackerNews/API).

SKlearn CountVectorizer is used to test on 'text' column to predict 'score'. Test accuracy rate using Multinomial, Bernoulli, Logistic Regression.

KNN framework where X featured: 'deleted', 'dead', 'descendants','poll', 'story', 'post_hour','post_day' to predict score.
