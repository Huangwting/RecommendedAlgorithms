import pandas as pd
from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split

# 读取数据
ratings = pd.read_csv('Data/ratings.csv')
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# 划分训练测试集
trainset, testset = train_test_split(data, test_size=0.3)

# SVD模型训练
algo = SVD(n_epochs=50, lr_all=0.01, reg_all=0.1)
algo.fit(trainset)


# 推荐函数
def recommend(user_id):
    unrated_movies = set(ratings['movieId']) - set(ratings[ratings['userId'] == user_id]['movieId'])

    predictions = []
    for movie_id in unrated_movies:
        prediction = algo.predict(uid=user_id, iid=movie_id)
        predictions.append(prediction)

    recommendations = [(pred.iid, pred.est) for pred in predictions]

    recommendations.sort(key=lambda x: x[1], reverse=True)

    return recommendations[:1000]


# 计算评价指标
def evaluate(recommendations, user_id):
    target_movies = [rating[1] for rating in
                     testset if rating[0] == user_id]
    # 只提取recommendations中的电影ID
    rec_movies = [rec[0] for rec in recommendations]

    hit = len(set(rec_movies) & set(target_movies))
    print(rec_movies)
    print(target_movies)
    precision = hit / len(recommendations)
    recall = hit / len(target_movies)
    hit_rate = hit / len(target_movies)

    return precision, recall, hit_rate

# 输出结果
user_id = 1
recommendations = recommend(user_id)[:10000]
print(recommendations)
precision, recall, hit_rate = evaluate(recommendations, user_id)
print("Precision: {:.3f}".format(precision))
print("Recall: {:.3f}".format(recall))
print("Hit rate: {:.3f}".format(hit_rate))