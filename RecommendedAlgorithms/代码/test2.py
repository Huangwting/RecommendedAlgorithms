from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#读取电影数据
movies_names = pd.read_csv('Data/movies.csv')

#读取评分数据
ratings = pd.read_csv('Data/ratings.csv')

#划分训练测试集
train, test = train_test_split(ratings, test_size=0.3, random_state=2)

#计算用户-电影评分矩阵，使用pivot方法创建用户电影评分矩阵，行表示用户，列表示电影，值表示用户评分，任何缺失的值都用零填充
matrix = train.pivot(index='userId', columns='movieId', values='rating').fillna(0)

#使用奇异值分解（SVD）进行降维
n_components = 10  #设置奇异值分解的维度，指定降维后要保留的数
svd = TruncatedSVD(n_components=n_components)
matrix_svd = svd.fit_transform(matrix) #将svd模型拟合到用户电影评级矩阵，并将矩阵转换到降维空间（matrix_swd）中

#基于svd变换后的用户电影评分矩阵来计算相关系数矩阵,计算用户相似度矩阵
user_sim = np.corrcoef(matrix_svd)

#计算电影的平均评分和流行度
average_ratings = train.groupby('movieId')['rating'].mean().values #计算训练集中每部电影的平均评分,结果是一个NumPy数组
movie_popularity = train.groupby('movieId')['rating'].count().values #计算训练集中每部电影的计数（流行度），结果是一个NumPy数组

#给用户推荐电影
#注释详见test1
def recommend_svd(user_id, k=10):
    recommendations = []
    similarity_scores = user_sim[user_id]
    similar_users = similarity_scores.argsort()[::-1][:k]

    movie_names_dict = movies_names.set_index('movieId')['title'].to_dict()

    for movie_id in matrix.loc[user_id].index:
        if matrix.loc[user_id][movie_id] == 0:
            avg_rating = (np.dot(user_sim[user_id, similar_users],
                                 matrix.loc[similar_users, movie_id])
                          / np.sum(np.abs(user_sim[user_id, similar_users])))
            movie_name = movie_names_dict.get(movie_id, 'Unknown')
            recommendations.append((movie_id, movie_name, avg_rating))

    recommendations.sort(key=lambda x: x[2], reverse=True)
    return recommendations

#输出推荐结果
user_id = 414
movies_list = recommend_svd(user_id, k=5)
movies_list = movies_list[:10]
print(movies_list)

#计算推荐指标
target_movies = test[test['userId'] == user_id]['movieId'].tolist()
hit = len((set(movie[0] for movie in movies_list)).intersection(set(target_movies)))
precision = hit / len(movies_list)
recall = hit / len(target_movies)
f1 = (2 * precision * recall) / (precision + recall) #计算F1分数
print('Precision=%.3f, Recall=%.3f, F1=%.3f' % (precision, recall, f1))


'''
# 可视化方法一：可视化用户相似性矩阵热图
#使用seaborn库,颜色强度表示用户之间的相似性
plt.figure(figsize=(10, 8))
sns.heatmap(user_sim, cmap='viridis', annot=False)
plt.title('User Similarity Matrix after Truncated SVD')
plt.xlabel('User ID')
plt.ylabel('User ID')
plt.show()
'''
#由于已将维度减少到10个分量，应用截断SVD后可视化缩减的维度空间。
#可视化降维空间，创建散点图
plt.figure(figsize=(8, 6))
plt.scatter(matrix_svd[:, 0], matrix_svd[:, 1], c='blue', alpha=0.5)
plt.title('Reduced-dimensional Space after Truncated SVD')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()
