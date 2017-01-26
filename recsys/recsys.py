import pandas as pd
import numpy as np

def s(user1, user2, user1average, user2average):
    global df

    movies_user1 = df[(df['userId'] == user1)]['movieId']
    movies_user2 = df[(df['userId'] == user2)]['movieId']

    # print sorted(movies_user1)
    # print sorted(movies_user2)

    intersection = []

    for movieid1 in movies_user1:
        for movieid2 in movies_user2:
            if movieid1 == movieid2:
                intersection.append(movieid1)

    # print "Intersection", intersection

    if len(intersection) == 0:
        # print "No Similarity"
        return 0

    sum_numerador = 0
    sum_denominador1 = 0
    sum_denominador2 = 0

    for i in intersection:
        # print "movie", i
        rating_user1 = float(df[(df['userId'] == user1) & (df['movieId'] == i)]['rating'])
        rating_user2 = float(df[(df['userId'] == user2) & (df['movieId'] == i)]['rating'])
        # print float(rating_user1), "User", user1, float(rating_user2), "User", user2
        # exit()
        sum_numerador += (rating_user1 - user1average) * (rating_user2 - user2average)
        sum_denominador1 += (rating_user1 - user1average)**2
        sum_denominador2 += (rating_user2 - user2average)**2

    return user1average + (sum_numerador / (sum_denominador1 + sum_denominador2))


df = pd.read_csv('ml-latest-small/ratings.csv')
# print df[(df['userId']==388) & (df['movieId']==6)]

users = df['userId'].unique()
movies = df['movieId'].unique()

target_user = 3
ratings = df[df['userId'] == target_user]

predictions = []
user_average = np.average(ratings['rating'])

for movie in movies:
    users_that_rate_this_movie = df[df['movieId'] == movie]
    soma_numerador = 0
    soma_denominador = 0

    # print users_that_rate_this_movie['userId']

    for user in users_that_rate_this_movie['userId']:
        current_user_average = np.average(df[(df['userId'] == user)]['rating'])
        sim = s(target_user, user, user_average, current_user_average)
        # print sim, "Similarity"
        # if not sim > 0:
        #     continue
        user_rating = float(df[(df['userId'] == user) & (df['movieId'] == movie)]['rating'])
        soma_numerador += (sim * (user_rating - current_user_average))
        soma_denominador += abs(sim)

    # if soma_denominador == 0:
    #     continue

    print movie
    predictions.append(user_average + soma_numerador / soma_denominador)
    print predictions
    # break

