#!/usr/bin/env python
# coding: utf-8

# # MovieLens Recomendation

# In[1]:


# importing necessary libraries
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from surprise import Dataset
from surprise import Reader
from surprise import SVD, SVDpp
from surprise.prediction_algorithms import KNNWithMeans, KNNBasic, KNNWithZScore
from surprise.model_selection import GridSearchCV
from surprise.model_selection import cross_validate

import ipywidgets as widgets
from ipywidgets import interact, interactive
from IPython.display import display, clear_output


# ## 加载数据

# ### Movies Data

# movie.csv：包含电影信息
# > movieId,title,genres
# * `movieId`: 每个电影的唯一标识
# * `title`: 电影的名称及其发布年份
# * `genres`: 电影可能属于的类别，用|分隔

# In[2]:


# movies data
# 包含9742部电影
# 需要把不同类别的电影分别分析
movies_df = pd.read_csv('Data/movies.csv')
print('Size of movies data:', movies_df.shape)
movies_df.head()


# In[3]:


# 无缺失数据
movies_df.info()


# ### Ratings Data

# ratings.csv:包含评分信息
# > userId,movieId,rating,timestamp
# * `userId`: 每个用户的唯一标识
# * `movieId`: 每个用户的唯一标识
# * `rating`: 评分
# * `timestamp`: 时间戳

# In[4]:


# ratings data
# 包含100836条评分
# timestamp用处不大
ratings_df = pd.read_csv('Data/ratings.csv')
print('Size of ratings data:', ratings_df.shape)
ratings_df.head()


# In[5]:


# 无缺失数据
ratings_df.info()


# ### Links Data

# links.csv:包含链接中的电影标识信息
# > movieId,imdbId,tmdbId
# * `movieId`: https://movielens.org 中电影的唯一标识
# * `imdbId`: http://www.imdb.com 中电影的唯一标识
# * `tmdbId`: https://www.themoviedb.org 中电影的唯一标识

# In[6]:


# Links data
# 9742部电影在三个数据源中的标识
links_df = pd.read_csv('Data/links.csv')
print('Size of links data:', links_df.shape)
links_df.head()


# In[7]:


links_df.info()


# ### Tags Data

# tags.csv:包含电影标签信息
# > userId,movieId,tag,timestamp
# * `userId`: 每个用户的唯一标识
# * `movieId`: 每个电影的唯一标识
# * `tag`: 用户对于电影的标签短语
# * `timestamp`: 时间戳

# In[8]:


# tags data
# 3683条标签信息
tags_df = pd.read_csv('Data/tags.csv')
print('Size of tags data:', tags_df.shape)
tags_df.head()


# In[9]:


tags_df.info()


# ## 数据分析

# In[10]:


# 删除ratings和tags中的timestamp
ratings_df.drop(columns='timestamp', inplace=True)
tags_df.drop(columns='timestamp', inplace=True)


# In[11]:


# 均值约为3.5
ratings_df['rating'].describe()


# In[12]:


ratings_df['rating'].value_counts()


# In[13]:


# 评分条形图
rating_counts = ratings_df['rating'].value_counts()
plt.figure(figsize=(16, 8))
sns.barplot(x=rating_counts.index, y=rating_counts.values, palette="viridis")
# 图表标题和轴标签
plt.title('Movie Rating Counts')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.savefig('Images/movie_rating')
plt.show()


# In[14]:


# 从电影标题中提取发行年份
movies_df['year'] = movies_df['title'].str.extract('.*\((.*)\).*',expand = False)


# In[15]:


# 异常值：2006-2007、NaN
movies_df['year'].unique()


# In[16]:


# 查找年份为'2006-2007'的电影
movies_df[movies_df['year'] == "2006–2007"]


# In[17]:


# 更改为2007
movies_df['year'] = movies_df['year'].replace("2006–2007","2007")


# In[18]:


# 查找没有年份信息的电影
movies_df[pd.isna(movies_df['year'])]


# In[19]:


# 删除没有年份信息和的电影
movies_df = movies_df.dropna(subset=['year'],how='any')


# In[20]:


# 将'year'列的数据类型转换为int
movies_df['year'] = movies_df['year'].astype(int)


# In[21]:


movies_df['year'].describe()


# In[22]:


movies_df['year'].value_counts()


# In[23]:


# 1902-2018年
# 评分条形图
movies_counts = movies_df['year'].value_counts()
plt.figure(figsize=(16, 8))
sns.barplot(x=movies_counts.index, y=movies_counts.values, palette="viridis")
# 图表标题和轴标签
plt.title('Movie Year Counts')
plt.xlabel('year')
plt.xticks(rotation=90)
plt.ylabel('Count')
plt.savefig('Images/movie_year')
plt.show()


# In[24]:


# 重复值检查
# movieId?
movies_df["movieId"].is_unique


# In[25]:


# title?
movies_df["title"].is_unique


# In[26]:


# 5部同名电影
movies_df[movies_df.duplicated(["title"], keep=False)].sort_values(by="title")


# In[27]:


# 替换重复电影标识
movie_id_change = {6003:144606, 838:26958, 32600:147002, 2851:168358, 34048:64997}
movies_df['movieId'].replace(movie_id_change,inplace=True)
movies_df = movies_df.drop_duplicates(subset=["movieId","title"])
ratings_df['movieId'].replace(movie_id_change,inplace=True)
tags_df['movieId'].replace(movie_id_change,inplace=True)
links_df['movieId'].replace(movie_id_change,inplace=True)


# In[28]:


# title?
movies_df["title"].is_unique


# In[29]:


# genre处理
movies_df['genres'] = movies_df['genres'].replace('(no genres listed)', np.nan)
movies_df['genres'].isna().sum()


# In[30]:


# 删除缺失值
movies_df = movies_df.dropna(subset=['genres'],how='any')
movies_df = movies_df.reset_index(drop=True)


# In[31]:


# 分隔类别
genres = []
for i in range(len(movies_df.genres)):
    for x in movies_df.genres[i].split('|'):
        if x not in genres:
            genres.append(x)
genres


# In[32]:


# 独热编码
for x in genres:
    movies_df[x] = 0
for i in range(len(movies_df.genres)):
    for x in movies_df.genres[i].split('|'):
        movies_df[x][i]=1


# In[33]:


# 删除genres列
movies_df.drop(columns='genres', inplace=True)
movies_df.sort_index(inplace=True)
movies_df


# In[34]:


# plotting genres popularity
x = {}
for i in movies_df.columns[4:23]:
 x[i] = movies_df[i].sum()
 print(f"{i:<15}{x[i]:>10}")
plt.figure(figsize=(16,8))
sns.barplot(x=list(x.keys()), y=list(x.values()), palette="viridis")
# 图表标题和轴标签
plt.title('Movie Genre Counts')
plt.xlabel('Genres')
plt.xticks(rotation=60, ha='right')
plt.ylabel('Count')
plt.savefig('Images/movie_genre')
plt.show()


# In[35]:


# 统计平均评分和评分人数
mean_rating = ratings_df.groupby('movieId').rating.mean().rename('mean rating')
num_rating = ratings_df.groupby('movieId').userId.count().rename('num rating')
# 加入表中
movies_df = pd.merge(movies_df, mean_rating, how='left', on='movieId')
movies_df = pd.merge(movies_df, num_rating, how='left', on='movieId')
# 缺失补0
movies_df['mean rating'].fillna(0, inplace=True)
movies_df['num rating'].fillna(0, inplace=True)

movies_df[['title', 'mean rating', 'num rating']]


# ## 基础推荐系统

# In[36]:


movie_ratings = movies_df[['title', 'mean rating', 'num rating', ]]


# In[37]:


# 最多评分电影
movie_ratings.sort_values(by=['num rating'], ascending=False).head(10)


# In[38]:


# 最高评分电影
movie_ratings.sort_values(by=['mean rating'], ascending=False).head(10)


# In[39]:


# 设置最低评分数量的阈值
minimum_num_ratings = 100
movie_ratings[movie_ratings['num rating']>minimum_num_ratings].sort_values(by=['mean rating'], ascending=False).head(10)


# In[40]:


# 评分次数较高的电影与平均评分较高的关联性较高
sns.scatterplot(data=movies_df, x='mean rating', y ='num rating');
plt.savefig('Images/rating_relevance')


# In[41]:


# 读者想要的特定类别
user_genre = 'Romance'
movie_ratings[movies_df[user_genre] == 1].sort_values(by=['mean rating'], ascending=False).head(10)


# In[42]:


# 设置最低评分数量的阈值
user_genre = 'Romance'
minimum_num_ratings = 100
movie_ratings[(movies_df[user_genre] == 1) & (movies_df['num rating']>minimum_num_ratings)].sort_values(by=['mean rating'], ascending=False).head(10)


# In[43]:


def naive_recommendation(threshold,ch_genre):
    
    minimum_num_ratings = threshold
    if ch_genre == 'All':
        result = movie_ratings[(movies_df['num rating']>minimum_num_ratings)].sort_values(by=['mean rating'], ascending=False).head(10)    
    else:
        result = movie_ratings[(movies_df[ch_genre] == 1) & (movies_df['num rating']>minimum_num_ratings)].sort_values(by=['mean rating'], ascending=False).head(10)
    
    print('\n\nRecommendations System')
    print('Minimum number of ratings:',threshold)
    print("Choice of genre:",ch_genre)
    display(result)

    
genres = ['All',
          'Animation',
          'Children',
          'Comedy',
          'Fantasy',
          'Romance',
          'Drama',
          'Action',
          'Crime',
          'Thriller',
          'Horror',
          'Mystery',
          'Sci-Fi',
          'War',
          'Musical',
          'Documentary',
          'IMAX',
          'Western',
          'Film-Noir'
         ]

w = interactive(naive_recommendation, threshold=widgets.IntSlider(min=0, max=200, value=100, step=5),
                       ch_genre=widgets.Dropdown(options=genres, description="Genre")
               )
display(w)


# ## 协同过滤

# In[44]:


df = pd.merge(ratings_df, movies_df, how='left', on = 'movieId')
# 创建矩阵：电影与用户
movie_user_matrix = df.pivot_table(index='userId', columns='title', values='rating')
movie_user_matrix


# In[45]:


movie_id = 900
movie_name = movies_df.loc[movies_df['movieId'] == movie_id, 'title'].values[0]
movie_name


# In[46]:


movie_ratings_df = movie_user_matrix[movie_name]
movie_ratings_df.head()


# In[47]:


correlation = movie_user_matrix.corrwith(movie_ratings_df)
similar_movies = pd.DataFrame(correlation, columns=['Correlation'])
# 删除缺失值
similar_movies.dropna(inplace=True)
similar_movies.sort_values('Correlation', ascending=False).head(10)


# In[48]:


# 设置最低评分数量的阈值
similar_movies = pd.merge(similar_movies, movies_df[['title','num rating']].drop_duplicates(), left_index=True, right_on='title')
similar_movies.set_index('title', inplace=True)
threshold = 50
similar_movies.sort_values('Correlation', ascending=False)[similar_movies['num rating']>threshold].head(10)


# ## surprise库

# KNN 算法：
# cv_knn_basic：使用 KNNBasic 算法进行交叉验证。这个算法是基于k最近邻的协同过滤算法，用于用户和物品之间的相似度计算。
# cv_knn_means：使用 KNNWithMeans 算法进行交叉验证。考虑了用户的平均评分。
# cv_knn_z：使用 KNNWithZScore 算法进行交叉验证。考虑了评分的z-score标准化。
# 矩阵分解算法：
# cv_svd：使用 SVD 算法进行交叉验证。这个算法是一种矩阵分解算法，用于将用户和物品的评分矩阵分解为潜在特征。
# cv_svd_pp：使用 SVDpp 算法进行交叉验证。包含了概率矩阵分解。

# In[49]:


# 训练集数据准备
reader = Reader(rating_scale=(0,5))
data = Dataset.load_from_df(ratings_df, reader)
dataset = data.build_full_trainset()
print('用户数量', dataset.n_users, '\n')
print('电影数量', dataset.n_items)


# In[50]:


# knn算法
cv_knn_basic = cross_validate(KNNBasic(), data, cv=5, n_jobs=5, verbose=True)
cv_knn_means = cross_validate(KNNWithMeans(), data, cv=5, n_jobs=5, verbose=True)
cv_knn_z = cross_validate(KNNWithZScore(), data, cv=5, n_jobs=5, verbose=True)


# In[51]:


# 矩阵分解算法
cv_svd = cross_validate(SVD(), data, cv=5, n_jobs=5, verbose=True)
cv_svd_pp = cross_validate(SVDpp(), data, cv=5, n_jobs=5, verbose=True)


# In[52]:


# Printing out the results for these algoritms
print('Evaluation Results:')
print('Algoritm\t RMSE\t\t MAE')
print()
print('KNN Basic', '\t', round(cv_knn_basic['test_rmse'].mean(), 4), '\t\t', round(cv_knn_basic['test_mae'].mean(), 4))
print('KNN Means', '\t', round(cv_knn_means['test_rmse'].mean(), 4), '\t', round(cv_knn_means['test_mae'].mean(), 4))
print('KNN ZScore', '\t', round(cv_knn_z['test_rmse'].mean(), 4), '\t\t', round(cv_knn_z['test_mae'].mean(), 4))
print()
print('SVD', '\t\t', round(cv_svd['test_rmse'].mean(), 4), '\t', round(cv_svd['test_mae'].mean(), 4))
print('SVDpp', '\t\t', round(cv_svd_pp['test_rmse'].mean(), 4), '\t', round(cv_svd_pp['test_mae'].mean(), 4))


# In[53]:


# 比较不同算法
# 选择 KNN ZScore 和 SVDpp
x_algo = ['KNN Basic', 'KNN Means', 'KNN ZScore', 'SVD', 'SVDpp',]
all_algos_cv = [cv_knn_basic, cv_knn_means, cv_knn_z, cv_svd, cv_svd_pp]

rmse_cv = [round(res['test_rmse'].mean(), 4) for res in all_algos_cv]
mae_cv = [round(res['test_mae'].mean(), 4) for res in all_algos_cv]

plt.figure(figsize=(20,5))

# RMSE
plt.subplot(1, 2, 1)
plt.title('Comparison of Algorithms on RMSE', loc='center', fontsize=15)
plt.plot(x_algo, rmse_cv, label='RMSE', color='red', marker='o')
plt.xlabel('Algorithms', fontsize=15)
plt.ylabel('RMSE Value', fontsize=15)
plt.legend()
plt.grid(ls='dashed')

# MAE
plt.subplot(1, 2, 2)
plt.title('Comparison of Algorithms on MAE', loc='center', fontsize=15)
plt.plot(x_algo, mae_cv, label='MAE', color='orange', marker='o')
plt.xlabel('Algorithms', fontsize=15)
plt.ylabel('MAE Value', fontsize=15)
plt.legend()
plt.grid(ls='dashed')
plt.savefig('Images/RMSE_MAE')
plt.show()


# ## KNN Based Algorithms

# I will now optimize on these two models. Let's start with KNN Means. We will optimize two hyperparameters: `k(numver of neighbors` and `distance metric`.First, I search for optimal `k` between 5 and 100.

# In[54]:


# 网格搜索最近邻数量
param_grid = {'k': [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100]}

gs_knn_zscore = GridSearchCV(KNNWithZScore, param_grid, measures=['rmse', 'mae'], cv=5, n_jobs=5)
gs_knn_zscore.fit(data)

y1 = gs_knn_zscore.cv_results['mean_test_rmse']
y2 = gs_knn_zscore.cv_results['mean_test_mae']


# In[55]:


# 绘图
plt.figure(figsize = (18, 15))

x = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100]

plt.subplot(1, 2, 1)
plt.title('K Neighbors vs RMSE', loc='center', fontsize=15)
plt.plot(x, y1, label='KNNWithZScore', color='red', marker='o')
plt.xlabel('K Neighbors', fontsize=15)
plt.ylabel('RMSE Value', fontsize=15)
plt.legend()
plt.grid(ls='dotted')

plt.subplot(1, 2, 2)
plt.title('K Neighbors vs MAE', loc='center', fontsize=15)
plt.plot(x, y2, label='KNNWithZScore', color='orange', marker='o')
plt.xlabel('K Neighbors', fontsize=15)
plt.ylabel('MAE Value', fontsize=15)
plt.legend()
plt.grid(ls='dotted')
plt.savefig('Images/K vs RMSE_MAE')
plt.show()


# In[56]:


# 最优k值为50
best_k1 = gs_knn_zscore.best_params['rmse']['k']
print("最优的k值:", best_k1)
best_k2 = gs_knn_zscore.best_params['mae']['k']
print("最优的k值:", best_k2)


# In[57]:


# 距离矩阵
knn_means_cosine = cross_validate(KNNWithZScore(k=50, sim_options={'name':'cosine'}), data, cv=5, n_jobs=5, verbose=True)
knn_means_pearson = cross_validate(KNNWithZScore(k=50, sim_options={'name':'pearson'}), data, cv=5, n_jobs=5, verbose=True)
knn_means_msd = cross_validate(KNNWithZScore(k=50, sim_options={'name':'msd'}), data, cv=5, n_jobs=5, verbose=True)
knn_means_pearson_baseline = cross_validate(KNNWithZScore(k=50, sim_options={'name':'pearson_baseline'}), data, cv=5, n_jobs=5, verbose=True)


x_distance = ['cosine', 'pearson', 'msd', 'pearson_baseline',]
all_distances_cv = [knn_means_cosine, knn_means_pearson, knn_means_msd, knn_means_pearson_baseline]

rmse_cv = [round(res['test_rmse'].mean(), 4) for res in all_distances_cv]
mae_cv = [round(res['test_mae'].mean(), 4) for res in all_distances_cv]

plt.figure(figsize=(20,5))

plt.subplot(1, 2, 1)
plt.title('Comparison of Distance Metrics on RMSE', loc='center', fontsize=15)
plt.plot(x_distance, rmse_cv, label='RMSE', color='red', marker='o')
plt.xlabel('Distance Metrics', fontsize=15)
plt.ylabel('RMSE Value', fontsize=15)
plt.legend()
plt.grid(ls='dashed')

plt.subplot(1, 2, 2)
plt.title('Comparison of Distance Metrics on MAE', loc='center', fontsize=15)
plt.plot(x_distance, mae_cv, label='MAE', color='orange', marker='o')
plt.xlabel('Distance Metrics', fontsize=15)
plt.ylabel('MAE Value', fontsize=15)
plt.legend()
plt.grid(ls='dashed')
plt.savefig('Images/Distance_metrics')
plt.show()


# KNN-ZScore with k=50 and pearson
# 
#  - **RMSE**: 0.8955
#  - **MAE** : 0.6765

# In[58]:


svd_param_grid = {'n_epochs': [20, 25, 30, 40, 50],
                  'lr_all': [0.007, 0.009, 0.01, 0.02],
                  'reg_all': [0.02, 0.04, 0.1, 0.2]}

gs_svd = GridSearchCV(SVD, svd_param_grid, measures=['rmse', 'mae'], cv=5, n_jobs=5)
gs_svd.fit(data)


# In[59]:


print('Best value for SVD  -RMSE:', round(gs_svd.best_score['rmse'], 4), '; MAE:', round(gs_svd.best_score['mae'], 4))
print('Optimal params RMSE =', gs_svd.best_params['rmse'])
print('optimal params MAE =', gs_svd.best_params['mae'])


# ### Predictions

# In[60]:


from surprise.model_selection import train_test_split
from surprise import accuracy

# 划分数据集为训练集和测试集
trainset, testset = train_test_split(data, test_size=0.3,random_state=42)

# 创建并训练 KNNWithZScore 模型
final_knn_model = KNNWithZScore(k=50, sim_options={'name': 'pearson'})
final_knn_model.fit(trainset)

# 在测试集上进行预测
predictions = final_knn_model.test(testset)

# 计算 RMSE 和 MAE
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)


# In[61]:


final_svdd_model = SVD(n_epochs=50, lr_all=0.01, reg_all=0.1)
final_svdd_model.fit(trainset)

# 在测试集上进行预测
predictions = final_svdd_model.test(testset)

# 计算 RMSE 和 MAE
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)


# In[62]:


final_svd_model = SVDpp(n_epochs=50, lr_all=0.01, reg_all=0.1)
final_svd_model.fit(trainset)

# 在测试集上进行预测
predictions = final_svd_model.test(testset)

# 计算 RMSE 和 MAE
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)


#  SVD 模型的 MAE 和 RMSE 更低

# In[138]:


def get_movie_recommendations(user_id, top_N=10):
    # 获取目标用户未评级的电影
    # 获取测试集中的所有电影
    # 筛选出测试集中属于目标用户的数据
    testset_movie = [item for item in testset]
    # 获取目标用户在测试集中观看的电影列表
    movie_list = [item[1] for item in testset_movie]

    # 为目标用户生成推荐列表
    recommendations = []
    for movie_id in movie_list:
        estimated_rating = final_svdd_model.predict(user_id, movie_id).est
        recommendations.append((movie_id, estimated_rating))

    # 根据预测评分降序排序推荐列表
    recommendations.sort(key=lambda x: x[1], reverse=True)

    # 获取前N部推荐电影
    top_recommendations = recommendations[:top_N]

    return top_recommendations


# In[161]:


# 输入目标用户ID，获取电影推荐
user_id = 555  # 替换为您的目标用户ID
top_N = 100  # 替换为您想要的推荐数量
recommendations = get_movie_recommendations(user_id, top_N)
print("用户", user_id, "的电影推荐：")
for movie_id, estimated_rating in recommendations:
    print("电影ID:", movie_id, "预测评分:", estimated_rating)


# In[162]:


# 筛选出测试集中属于目标用户的数据
testset_for_user = [item for item in testset if item[0] == user_id]
# 获取目标用户在测试集中观看的电影列表
target_movie_list = [item[1] for item in testset_for_user]
target_movie_list


# In[163]:


# 计算推荐命中次数
hit = sum(1 for movie_id, _ in recommendations if movie_id in target_movie_list)
hit


# In[164]:


precision = hit / len(recommendations)
recall = hit / len(target_movie_list)
print('Precision=%.3f, Recall=%.3f' % (precision, recall))


# ## 混合推荐引擎

# In[165]:


def hybrid_recommendation_engine(user_id='new',preferred_genre='all',minimum_num_ratings=50):
    
    if user_id=='new':
        if preferred_genre == 'all':
            result = movie_ratings[(movies_df['num rating']>minimum_num_ratings)].sort_values(by=['mean rating'], ascending=False).head(10)    
        else:
            result = movie_ratings[(movies_df[preferred_genre] == 1) & (movies_df['num rating']>minimum_num_ratings)].sort_values(by=['mean rating'], ascending=False).head(10)

    else:
        new_df = df.copy()
    
        # 选择类别
        if preferred_genre !='all':
            new_df = new_df[new_df[preferred_genre]==1]

        # 评分数量过滤
        new_df = new_df[new_df['num rating']>=minimum_num_ratings]

        # 已评分电影过滤
        movies_already_watched = set(new_df[new_df['userId']==user_id].movieId.values)
        new_df= new_df[~new_df['movieId'].isin(movies_already_watched)]

        # 预测评分
        all_movie_ids = set(new_df['movieId'].values)
        all_movie_ratings = []

        for i in all_movie_ids:
            expected_rating = final_svdd_model.predict(uid=user_id, iid=i).est
            all_movie_ratings.append((i,round(expected_rating,1)))

        # 过滤出结果
        expected_df = pd.DataFrame(all_movie_ratings, columns=['movieId','Expected Rating'])    
        result = pd.merge(expected_df, movies_df[['movieId','title','num rating']],on='movieId')
        result = result.sort_values(['Expected Rating','num rating'],ascending=[False,False])
        result = result.head()
    
    
    print('\n\nRecommendations System')
    print('User id:',user_id)
    print('Minimum number of ratings:',minimum_num_ratings)
    print("Choice of genre:", preferred_genre)
    display(result)


        
genres = ['all',
          'Animation',
          'Children',
          'Comedy',
          'Fantasy',
          'Romance',
          'Drama',
          'Action',
          'Crime',
          'Thriller',
          'Horror',
          'Mystery',
          'Sci-Fi',
          'War',
          'Musical',
          'Documentary',
          'IMAX',
          'Western',
          'Film-Noir'
         ]
all_userids = ['new'] + list(set(df.userId.values))
w = interactive(hybrid_recommendation_engine,
                user_id=widgets.Dropdown(options=all_userids, description="user_id"),
                minimum_num_ratings=widgets.IntSlider(min=0, max=200, value=100, step=5),
                preferred_genre=widgets.Dropdown(options=genres, description="Genre")
               )
display(w)


# In[ ]:




