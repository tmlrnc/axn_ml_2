"""
This module illustrates how to retrieve the top-10 items with highest rating
prediction. We first train an SVD algorithm on the MovieLens dataset, and then
predict all the ratings for the pairs (user, item) that are not in the training
set. We then retrieve the top-10 prediction for each user.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from collections import defaultdict

from surprise import SVD
from surprise import Dataset


import pandas as pd

from surprise import NormalPredictor
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate

def get_top_n(predictions, n=10):
    """Return the top-N recommendation for each user from a set of predictions.
    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.
    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n



# Creation of the dataframe.
ratings_dict2 = {'itemID': [1, 7, 1, 2, 2, 3,3,3,3, 1],
                'userID': [1, 2, 2, 4, 1, 3, 4, 4, 4, 1],
                'rating': [3, 2, 4, 3, 1,3,2,3,3,1]}


# Creation of the dataframe.
ratings_dict3 = {'itemID': [1, 7, 7, 2, 2, 3,3,3,3, 1],
                'userID': [1, 2, 2, 4, 1, 3, 4, 4, 4, 1],
                'rating': [1, 1, 1, 1, 1,1,1,1,1,1]}




import json
file_in_name1 = "/Users/tomlorenc/Downloads/results-20201218-075350.json"
file_in_name2 = "/Users/tomlorenc/Downloads/results-20201218-091546.json"
file_in_name3 = "/Users/tomlorenc/Downloads/results-20201218-091903.json"
file_in_name4 = "/Users/tomlorenc/Downloads/results-20201218-093345.json"
file_in_name6 = "/Users/tomlorenc/Downloads/results-20201218-101736.json"
file_in_name5 = "/Users/tomlorenc/Downloads/results-20201218-094845.json"




# read file
with open(file_in_name2, 'r') as myfile:
    data_json2=myfile.read()

# parse file
obj_json2 = json.loads(data_json2)

print(obj_json2)



# read file
with open(file_in_name1, 'r') as myfile:
    data_json1=myfile.read()

# parse file
obj_json1 = json.loads(data_json1)

print(obj_json1)




# read file
with open(file_in_name3, 'r') as myfile:
    data_json3=myfile.read()

# parse file
obj_json3 = json.loads(data_json3)

print(obj_json3)


with open(file_in_name4, 'r') as myfile:
    data_json4=myfile.read()

# parse file
obj_json4 = json.loads(data_json4)


with open(file_in_name5, 'r') as myfile:
   data_json5=myfile.read()

# parse file
obj_json5 = json.loads(data_json5)


with open(file_in_name6, 'r') as myfile:
   data_json6=myfile.read()

# parse file
obj_json6 = json.loads(data_json6)

obj_json = obj_json1 + obj_json2 + obj_json3 + obj_json4 + obj_json5 + obj_json5

print(obj_json)


ratings_dict = {}

itemID_list = []
userID_list = []
rating_list = []

ratings_dict = {'itemID': [], 'userID': [], 'rating': []}
ratings_dict_5 = {'itemID': [], 'userID': [], 'rating': []}

# show values
row_count = 0
for row in obj_json:
    print ("*****" )
    row_count = row_count + 1
    print ("row " + str(row) )
    print ("*****" )

    itemID = row['sid']

    print(type(itemID))
    print("sid " + str(itemID))

    if itemID is None:
        print("SKIP")
        continue

    print ("*****" )

    userID = row['uid']
    print("userID " + str(userID))

    if userID is None:
        print("SKIP")
        continue

    itemID_list.append(itemID)
    userID_list.append(userID)
    rating_list.append(1)

    ratings_dict_5['itemID'].append(itemID)
    ratings_dict_5['userID'].append(userID)
    ratings_dict_5['rating'].append(1)

# rememebr to read MOST POPULAR FOR COLD START
#print ("*****" )
ratings_dict['itemID'].append(itemID_list)
ratings_dict['userID'].append(userID_list)
ratings_dict['rating'].append(rating_list)

#print(ratings_dict_5)
#print(len(ratings_dict_5))



#print(ratings_dict)
#print(len(ratings_dict))
# Creation of the dataframe.
ratings_dict_test = {'itemID': [1, 7, 7, 2, 2, 3,3,3,3, 1],
                'userID': [1, 2, 2, 4, 1, 3, 4, 4, 4, 1],
                'rating': [1, 1, 1, 1, 1,1,1,1,1,1]}
#print(type(ratings_dict))
#print(type(ratings_dict))


#print(ratings_dict_test)
#print(len(ratings_dict_test))
df = pd.DataFrame(ratings_dict_5)

# A reader is still needed but only the rating_scale param is requiered.
reader = Reader(rating_scale=(1, 1))

# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
trainset = data.build_full_trainset()

algo = SVD()
algo.fit(trainset)

# Than predict ratings for all pairs (u, i) that are NOT in the training set.
testset = trainset.build_anti_testset()
predictions = algo.test(testset)

top_n = get_top_n(predictions, n=10)
print ("PRINT TOP 10 STORY IDs per USERID" )

# Print the recommended items for each user
for uid, user_ratings in top_n.items():
    print("******************")
    print(uid, [iid for (iid, _) in user_ratings])


print ("ROW COUNT *****" + str(row_count))

