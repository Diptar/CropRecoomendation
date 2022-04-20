import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv("Crop_recommendation.csv")
data["Grouped Data"] = data["N"].astype("str") + " " + data["P"].astype("str") + " " + data["K"].astype("str") + " " + data["temperature"].astype("str") + " " + data["humidity"].astype("str") + " " + data["ph"].astype("str") + " " + data["rainfall"].astype("str")

# Creating Count Vectors:
cv = CountVectorizer(max_features = 5000)
vectors = cv.fit_transform(data["Grouped Data"]).toarray()
similarity = cosine_similarity(vectors)
sorted(list(enumerate(similarity[0])),reverse = True,key = lambda x : x[1])

def get_names_of_crops(index):
    return data.iloc[index]["label"]

def Recommend(crop):
    crop_index = data[data["label"] == crop].index[0]
    distances = similarity[crop_index]
    ans = sorted(list(enumerate(distances)),reverse = True,key = lambda x : x[1])
 
    name = []
    for i in ans[:18]:
        name.append(get_names_of_crops(i[0]))
    name.pop(0)
    return name
