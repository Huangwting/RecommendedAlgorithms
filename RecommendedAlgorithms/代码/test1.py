from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt

#读取电影数据
movies_names = pd.read_csv('Data/movies.csv')

#读取评分数据
ratings = pd.read_csv('Data/ratings.csv')

#划分训练测试集
train, test = train_test_split(ratings, test_size=0.3, random_state=2)

#计算相似度矩阵
def calculate_similarity_matrix(train):
    #使用pivot方法将评分数据转换成用户-电影的评分矩阵。行表示用户，列表示电影，值表示用户评分，fillna（0）用于用零填充任何缺失的值
    matrix = train.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    #根据用户的电影评分计算用户之间的余弦相似性矩阵
    user_sim = cosine_similarity(matrix)
    #返回两个矩阵matrix（用户电影评分矩阵）和user_sim（用户相似性矩阵）
    return matrix, user_sim

#使用训练数据调用先前定义的calculate_simility_matrix函数，返回matrix（用户电影评分矩阵）和user_sim（用户相似性矩阵）
#得到训练集的用户-电影评分矩阵和用户相似度矩阵
matrix, user_sim = calculate_similarity_matrix(train)

#计算电影的平均评分和流行度
average_ratings = train.groupby('movieId')['rating'].mean().values #计算训练集中每部电影平均评分，结果是一个NumPy数组
movie_popularity = train.groupby('movieId')['rating'].count().values #计算训练集中每部电影的计数（流行度），结果是一个NumPy数组

#给用户推荐电影
def recommend(user_id, k=10):
    recommendations = [] #创建一个空列表,用于存储向用户的推荐结果
    similarity_scores = user_sim[user_id] #获取用户与其他用户的相似度得分
    similar_users = similarity_scores.argsort()[::-1][:k] #获取相似度最高的k个用户id

    movie_names_dict = movies_names.set_index('movieId')['title'].to_dict() #创建电影id到电影名称的映射字典

    for movie_id in matrix.loc[user_id].index: #遍历该用户未打分的电影id
        if matrix.loc[user_id][movie_id] == 0: #过滤掉用户已rating的电影
            #根据与其他对该电影进行分级的用户的相似性来计算预测avg_rating
            avg_rating = (np.dot(matrix.loc[similar_users, movie_id],
                                 average_ratings[similar_users] * movie_popularity[similar_users])
                          / np.sum(average_ratings[similar_users] * movie_popularity[similar_users]))
            movie_name = movie_names_dict.get(movie_id, 'Unknown') #获取电影名称
            recommendations.append((movie_id, movie_name, avg_rating)) #将结果(id,名称,得分)加入推荐列表

    recommendations.sort(key=lambda x: x[2], reverse=True) #按预测评分降序排序
    return recommendations #返回推荐结果列表

#输出推荐结果
user_id = 414 #指定用户
movies_list = recommend(user_id, k=15) #调用recommendance函数来获取指定用户的电影推荐
movies_list = movies_list[:10] #从生成的列表中检索前10个推荐
print(movies_list)

#计算推荐指标
target_movies = test[test['userId'] == user_id]['movieId'].tolist() #从测试集中检索指定用户（user_id）实际评分的电影列表
hit = len((set(movie[0] for movie in movies_list)).intersection(set(target_movies))) #计算与用户在测试集中的实际评分相匹配的推荐电影数
precision = hit / len(movies_list) #计算精确率，即正确推荐的电影与推荐电影总数的比率
recall = hit / len(target_movies) #计算召回率，即正确推荐的电影与用户在测试集中实际评分的电影总数的比率
f1 = (2 * precision * recall) / (precision + recall) #计算F1分数
print('Precision=%.3f, Recall=%.3f, F1=%.3f' % (precision, recall, f1))

#可视化建议的功能
#水平条形图，其中每个条形图代表一部推荐的电影，条形图的长度与平均预测评级相对应。y轴反转以提高可读性
def visualize_recommendations(movies_list):
    movie_titles = [movie[1] for movie in movies_list]
    scores = [movie[2] for movie in movies_list]

    plt.figure(figsize=(10, 6))
    plt.barh(movie_titles, scores, color='skyblue')
    plt.xlabel('Average Predicted Rating')
    plt.title('Top 10 Movie Recommendations for User {}'.format(user_id))
    plt.gca().invert_yaxis()  #反转y轴以获得更好的可读性
    plt.show()

visualize_recommendations(movies_list)