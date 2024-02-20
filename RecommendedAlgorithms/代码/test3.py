import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#读取电影数据
movies = pd.read_csv('Data/movies.csv')

#读取评分数据
ratings = pd.read_csv('Data/ratings.csv')

#将电影数据与评分数据合并
merged_data = pd.merge(ratings, movies, on='movieId')

#将数据集分割为训练集和测试集
train, test = train_test_split(merged_data, test_size=0.3, random_state=2)

#计算用户相似度矩阵
def calculate_user_similarity_matrix(train):
    #使用pivot方法将评分数据转换成用户-电影的评分矩阵。行表示用户，列表示电影，值表示用户评分，fillna（0）用于用零填充任何缺失的值
    user_movie_matrix = train.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    #根据用户的电影评分计算用户之间的余弦相似性矩阵
    user_sim = cosine_similarity(user_movie_matrix)
    #返回两个矩阵user_movie_matrix（用户电影评分矩阵）和user_sim（用户相似性矩阵）
    return user_movie_matrix, user_sim

#使用训练数据调用先前定义的calculate_simility_matrix函数
#得到训练集的用户-电影评分矩阵和用户相似度矩阵
user_movie_matrix, user_sim = calculate_user_similarity_matrix(train)

#根据电影类型计算电影相似度矩阵
def calculate_genre_similarity_matrix(movies):
    #将电影类型进行独热编码，每个电影的类型将被表示为一个二进制向量，其中每个元素表示一个可能的类型
    genres = movies['genres'].str.get_dummies('|')
    #计算电影类型相似度矩阵
    genre_sim = cosine_similarity(genres)
    #返回独热编码矩阵和基于电影类型计算的相似度矩阵。
    return genres, genre_sim

genres, genre_sim = calculate_genre_similarity_matrix(movies)

#组合协同过滤和基于内容的过滤
def hybrid_recommendation(user_id, k=10):
    recommendations = [] #创建一个空列表，用于存储最终的推荐结果
    user_similarity_scores = user_sim[user_id] #获取用户与其他用户的相似度得分
    similar_users = user_similarity_scores.argsort()[::-1][:k] #获取相似度最高的k个用户id

    for movie_id in user_movie_matrix.loc[user_id].index:
        if user_movie_matrix.loc[user_id][movie_id] == 0:
            #提取当前电影的类型信
            movie_genres = genres.loc[genres.index == movie_id]

            #确保电影具有类型信息
            if not movie_genres.empty:
                #计算基于内容的分数
                content_based_score = np.sum(user_movie_matrix.loc[similar_users, movie_id] * genre_sim[similar_users, movie_id])
                #计算协同过滤分数
                collaborative_filtering_score = np.dot(user_similarity_scores[similar_users],
                                                       user_movie_matrix.loc[similar_users, movie_id])

                #加权得分
                weighted_score = 0.75 * collaborative_filtering_score + 0.25 * content_based_score
                recommendations.append((movie_id, movies.loc[movies['movieId'] == movie_id, 'title'].values[0], weighted_score))

    recommendations.sort(key=lambda x: x[2], reverse=True) #按加权得分降序排序
    return recommendations[:k] #返回前k个推荐结果


# 输出混合推荐结果
user_id = 414
hybrid_movies_list = hybrid_recommendation(user_id, k=15)
hybrid_movies_list = hybrid_movies_list[:10]
print(hybrid_movies_list)

# 计算混合推荐指标
target_movies = test[test['userId'] == user_id]['movieId'].tolist()
hit = len((set(movie[0] for movie in hybrid_movies_list)).intersection(set(target_movies)))
precision = hit / len(hybrid_movies_list)
recall = hit / len(target_movies)
f1 = (2 * precision * recall) / (precision + recall) #计算F1分数
print('Precision=%.3f, Recall=%.3f, F1=%.3f' % (precision, recall, f1))

#可视化混合推荐的功能
#条形图表示为每部推荐电影分配的分数。
def visualize_recommendations(recommendations):
    movie_titles = [movie[1] for movie in recommendations]
    scores = [movie[2] for movie in recommendations]

    plt.figure(figsize=(12, 6))
    sns.barplot(x=scores, y=movie_titles, palette="viridis")
    plt.title('Hybrid Recommendations for User {}'.format(user_id))
    plt.xlabel('Weighted Score')
    plt.ylabel('Movie Title')
    plt.show()

visualize_recommendations(hybrid_movies_list)