from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.interactiveshell import InteractiveShell
%matplotlib inline
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode,iplot,plot
init_notebook_mode(connected=True)
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
# import keras
# import sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import GridSearchCV


## Data Preprocessing

#loading data
data = pd.read_csv("movie_metadata.csv")

data.shape
data.info()

#Summary of missing values
data.isnull().sum()

#replacing NaN in Numeric variables by mean value
data = data.fillna(data.mean())

#replacing NaN in character variables by mode value
for column in data[['color', 'language', 'country', 'content_rating']]:
    mode = data[column].mode()
    data[column] = data[column].fillna(mode)

#dropping remaining records having NaN
data = data.dropna(how='any')


#Summary of missing values
data.isnull().sum()
# data.info()



## Data Exploration and Visualization

pd.DataFrame.hist(data, figsize = [15,15]);


#histogram of imdb scores
data['imdb_score'].diff().hist()
plt.title("imdb_score")
plt.show()

#boxplot and scatter plots
meanpointprops = dict(marker = 'D', markeredgecolor = 'black',
                      markerfacecolor = 'firebrick')
meanlineprops = dict(linestyle='--', linewidth=2.5, color='purple')


#imdb score vs color
df = data[['color','imdb_score']]
df.boxplot(by = 'color',showmeans=True, figsize = [20,10])
plt.show()


#imdb score vs title word count
#create a new column title_words as the word count of the title
data['title_words'] = data['movie_title'].apply(lambda x: len(str(x).split(' ')))
print(data.head(4))
df = data[['title_words','imdb_score']]
df.boxplot(by = 'title_words',showmeans=True, figsize = [20,10])
plt.show()


#imdb score vs duration
df = data[['duration','imdb_score']]
df.boxplot(by='duration', showmeans=True, meanprops=meanpointprops, meanline=True, figsize = [20,10])
plt.show()


#imdb score vs country
df = data[['country','imdb_score']]
df.boxplot(by='country', showmeans=True, figsize = [20,10])
plt.show()


# #imdb score vs language
df = data[['language','imdb_score']]
df.boxplot(by='language', showmeans=True, figsize = [20,10])
plt.show()


#imdb score vs content rating
df = data[['content_rating','imdb_score']]
df.boxplot(by='content_rating', showmeans=True, figsize = [20,10])
plt.show()


#imdb score vs face number in the poster
df = data[['facenumber_in_poster','imdb_score']]
df.boxplot(by='facenumber_in_poster', showmeans=True, meanprops=meanpointprops, figsize = [20,10])
plt.show()
data.plot.scatter('imdb_score','facenumber_in_poster')


#imdb score vs title year
df = data[['title_year','imdb_score']]
df.boxplot(by='title_year', showmeans=True, figsize = [20,10])
plt.show()

#imdb score vs budget, gross and profit
data.plot.scatter('imdb_score','budget')
data.plot.scatter('imdb_score','gross')
data['profit'] = data['gross'] - data['budget']
data.plot.scatter('imdb_score','profit')


#imdb score vs facebook popularity
data.plot.scatter('imdb_score','movie_facebook_likes')
data.plot.scatter('imdb_score','cast_total_facebook_likes')
data.plot.scatter('imdb_score','director_facebook_likes')
data.plot.scatter('imdb_score','actor_1_facebook_likes')
data.plot.scatter('imdb_score','actor_2_facebook_likes')
data.plot.scatter('imdb_score','actor_3_facebook_likes')
# # plt.show()
#
#
#imdb score vs user reviews and critic reviews
data.plot.scatter('imdb_score','num_user_for_reviews')
data.plot.scatter('imdb_score','num_critic_for_reviews')
# # plt.show()
#
#voted users vs user reviews
data.plot.scatter('num_user_for_reviews','num_voted_users')


#correlation analysis
import seaborn
seaborn.heatmap(data.corr(), annot=True)

#dropping 'cast_total_facebook_likes' as 'cast_total_facebook_likes' and'actor_1_facebook_likes' are highly correlated
data = data.drop('cast_total_facebook_likes', axis=1)
#dropping 'num_user_for_reviews' as 'num_user_for_reviews'and 'num_voted_users' are highly correlated
data = data.drop('num_user_for_reviews', axis=1)

# seaborn.heatmap(data.corr(), annot=True)

# Summary:
# Distribution of IMDB score is left-skewed. Mean is lower than median. There are some movies really bad.
# There are much more color movies than black movies. Black movies have higher IMDB scores comparing to color movies.
# Most of the movie titles contain fewer than 7 words.
# Long movies tend to have high rating. Some really short movies also have higher rating.
# Most of the movie posters contain fewer than 5 faces.
# US and UK produced the greatest numbers of movies.
# Movies in recent years have lower and lower IMDB mean score.
# cast_total_facebook_likes and actor_1_facebook_likes are correlated.
# num_user_for_reviews and num_voted_users are correlated.



## Model Development and Evaluation

#select variables
data['color'] = data['color'].astype('category')
data['language'] = data['language'].astype('category')
data['country'] = data['country'].astype('category')
data['content_rating'] = data['content_rating'].astype('category')
# data.info()


# columns and training/test sets preparation
feature_cols = ['color','num_critic_for_reviews','duration','director_facebook_likes',
                    'actor_3_facebook_likes','actor_1_facebook_likes','gross','num_voted_users',
                    'facenumber_in_poster','language','country','content_rating','budget','profit',
                    'title_year','actor_2_facebook_likes','aspect_ratio','movie_facebook_likes']

numeric_cols = ['num_critic_for_reviews','duration','director_facebook_likes',
                    'actor_3_facebook_likes','actor_1_facebook_likes','gross',
                    'num_voted_users','facenumber_in_poster','budget','profit',
                    'title_year','actor_2_facebook_likes','aspect_ratio','movie_facebook_likes']
cat_cols = ['color','language','country','content_rating']


X = data.loc[:, numeric_cols]
y = data['imdb_score']
X = X.as_matrix().astype(np.float)
y = y.as_matrix().astype(np.float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=43)

#Linear regression
lm = LinearRegression()
lm.fit(X_train, y_train)

print('Coefficients: ', lm.coef_)
print('Intercept: ', lm.intercept_)

y_pred = lm.predict(X_test)
print("Mean squared error of linear regression: %.2f"
      % mean_squared_error(y_test, y_pred))
print('Variance score of linear regression: %.2f' % r2_score(y_test, y_pred))

fig, axes = plt.subplots()

axes.scatter(y_test, y_pred, color='red', alpha=0.5)
axes.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', lw=1)
axes.set_xlabel('Actual')
axes.set_ylabel('Predicted')


#conclusion: linear regression with the numeric variables is not a good model.

#Keras regression
data_k = data[['num_critic_for_reviews','duration','director_facebook_likes',
                    'actor_3_facebook_likes','actor_1_facebook_likes','gross','num_voted_users',
                    'facenumber_in_poster','budget','profit','title_year',
                    'actor_2_facebook_likes','aspect_ratio','movie_facebook_likes','imdb_score']]
X = data_k.iloc[:, 0:14].values
Y = data_k.iloc[:, 14:15].values

X.shape
Y.shape

#define base model
def baseline_model(learn_rate=0.01, momentum=0):
    model = Sequential()
    model.add(Dense(14, input_dim=14, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='relu'))
    optimizer = SGD(lr=learn_rate, momentum=momentum)
    model.compile(loss='mean_absolute_error', optimizer='adam')
    # model.compile(loss='mean_squared_error', optimizer='adam')
    return model


#evaluate base model
np.random.seed(42)
#evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=10, batch_size=50, verbose=1)
#sing 10-fold cross validation to evaluate the model
kfold = KFold(n_splits=10, random_state=42)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Results of keras regression: %.2f (std %.2f) MAE" % (results.mean(), results.std()))


#define larger model: adding more hidden layers to the base model
def larger_model(learn_rate=0.01, momentum=0):
    model = Sequential()
    model.add(Dense(12, input_dim=14, kernel_initializer='normal', activation='relu'))
    model.add(Dense(9, input_dim=14, kernel_initializer='normal', activation='relu'))
    model.add(Dense(7, input_dim=14, kernel_initializer='normal', activation='relu'))
    model.add(Dense(4, input_dim=14, kernel_initializer='normal', activation='relu'))
    model.add(Dense(2, input_dim=14, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='relu'))
    optimizer = SGD(lr=learn_rate, momentum=momentum)
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model

#evaluate larger model
np.random.seed(42)
estimator = KerasRegressor(build_fn=larger_model, nb_epoch=10, batch_size=50, verbose=1)
kfold = KFold(n_splits=10, random_state=42)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Keras results with more hidden layers: %.2f (std %.2f) MAE" % (results.mean(), results.std()))


#define wider model, nearly doubling the number of neurons in two hidden layers
def wider_model(learn_rate=0.01, momentum=0):
    model = Sequential()
    model.add(Dense(26, input_dim=14, kernel_initializer='normal', activation='relu'))
    model.add(Dense(30, input_dim=14, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='relu'))
    optimizer = SGD(lr=learn_rate, momentum=momentum)
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model


#evaluate wider model
np.random.seed(42)
estimator = KerasRegressor(build_fn=wider_model, nb_epoch=10, batch_size=50, verbose=1)
kfold = KFold(n_splits=10, random_state=42)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Keras results with more neurons: %.2f (std %.2f) MAE" % (results.mean(), results.std()))

# conclusion:
# Results of Keras regression: mean -6.44 (std 0.11) MAE
# Keras results with more hidden layers: -5.67 (std 1.05) MAE
# Keras results with more neurons: -6.44 (std 0.11) MAE
# Model with more number of hidden layers is performing better than the model having more number of neurons.


#Tensorflow wide & deep learning

color = tf.feature_column.categorical_column_with_vocabulary_list(
    'color', ['Black and White', 'Color'])
language = tf.feature_column.categorical_column_with_hash_bucket(
    'language', hash_bucket_size=1000)
content_rating = tf.feature_column.categorical_column_with_vocabulary_list(
    'content_rating', ['Approved', 'G', 'GP', 'M', 'NC-17', 'Not Rated', 'PG', 'PG-13',
              'Passed', 'R', 'TV-14', 'TV-G', 'TV-PG', 'Unrated', 'X'])
country = tf.feature_column.categorical_column_with_hash_bucket(
    'country', hash_bucket_size=1000)


title_year = tf.feature_column.numeric_column('title_year')
duration = tf.feature_column.numeric_column('duration')
gross = tf.feature_column.numeric_column('gross')
budget = tf.feature_column.numeric_column('budget')
profit = tf.feature_column.numeric_column('profit')
aspect_ratio = tf.feature_column.numeric_column('aspect_ratio')
facenumber_in_poster = tf.feature_column.numeric_column('facenumber_in_poster')
movie_facebook_likes = tf.feature_column.numeric_column('movie_facebook_likes')
director_facebook_likes = tf.feature_column.numeric_column('director_facebook_likes')
actor_1_facebook_likes = tf.feature_column.numeric_column('actor_1_facebook_likes')
actor_2_facebook_likes = tf.feature_column.numeric_column('actor_2_facebook_likes')
actor_3_facebook_likes = tf.feature_column.numeric_column('actor_3_facebook_likes')
num_critic_for_reviews = tf.feature_column.numeric_column('num_critic_for_reviews')
num_voted_users = tf.feature_column.numeric_column('num_voted_users')


base_columns = [
    title_year, duration, gross, budget, profit, aspect_ratio, facenumber_in_poster, movie_facebook_likes,
    director_facebook_likes, actor_1_facebook_likes, actor_2_facebook_likes, actor_3_facebook_likes,
    num_critic_for_reviews, num_voted_users, color, country, language, content_rating,
]


crossed_columns = [
    tf.feature_column.crossed_column(
        ['country', 'language'], hash_bucket_size=1000),
]

deep_columns = [
    num_critic_for_reviews,
    duration,
    director_facebook_likes,
    actor_1_facebook_likes,
    actor_2_facebook_likes,
    actor_3_facebook_likes,
    movie_facebook_likes,
    gross,
    budget,
    profit,
    num_voted_users,
    facenumber_in_poster,
    title_year,
    aspect_ratio,
    tf.feature_column.indicator_column(color),
    tf.feature_column.indicator_column(country),
    tf.feature_column.indicator_column(language),
    tf.feature_column.embedding_column(content_rating, dimension=14),
]


data_t = data[['color','num_critic_for_reviews','duration','director_facebook_likes',
                    'actor_3_facebook_likes','actor_1_facebook_likes','gross','num_voted_users',
                    'facenumber_in_poster','language','country','content_rating','budget','profit',
                    'title_year','actor_2_facebook_likes','aspect_ratio','movie_facebook_likes','imdb_score']]

train, test = train_test_split(data_t, test_size=0.2)


def input_fn(df, train= True):
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values)
                     for k in numeric_cols}
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {k: tf.SparseTensor(
        indices=[[i, 0] for i in range(df[k].size)],
        values=df[k].values,
        dense_shape=[df[k].size, 1])
                      for k in cat_cols}
    # Merges the two dictionaries into one.
    feature_columns = continuous_cols.copy()
    feature_columns.update(categorical_cols)
    # Converts the label column into a constant Tensor.
    y = tf.constant(df['imdb_score'].values)
    # Returns the feature columns and the label.
    return feature_columns, y

def train_input_fn():
  return input_fn(train)

def eval_input_fn():
  return input_fn(test, train = False)


estimator = tf.estimator.DNNLinearCombinedRegressor(
        model_dir='./tmp/model',
        linear_feature_columns=base_columns + crossed_columns,
        linear_optimizer=tf.train.FtrlOptimizer(
            learning_rate=1e-8,
            l1_regularization_strength=0.001,
            l2_regularization_strength=0.001),
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[1000, 500, 100],
        dnn_optimizer=tf.train.ProximalAdagradOptimizer(
            learning_rate=1e-8,
            l1_regularization_strength=0.001,
            l2_regularization_strength=0.001))


estimator.train(input_fn=train_input_fn, steps=50)
print(estimator.predict(input_fn=lambda: input_fn(test)))
results = estimator.evaluate(input_fn=eval_input_fn, steps=10)
for key in sorted(results):
    print("%s: %s" % (key, results[key]))

predict = estimator.predict(input_fn=eval_input_fn)
out = list(zip(list(data['movie_title']),list(results)))
cols = ['movie_title', 'imdb_score']
df_out = pd.DataFrame(out, columns=cols)

df_out.to_csv(path_or_buf='./tmp/pred.csv', index=False)