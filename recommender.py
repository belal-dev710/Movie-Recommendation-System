import pandas as pd
import numpy as np
import ast

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

movies = movies.merge(credits, on="title")

movies = movies[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]
movies = movies.dropna()

def get_names(x):
    result = []
    for item in ast.literal_eval(x):
        result.append(item["name"])
    return result

def get_top_cast(x):
    result = []
    count = 0
    for item in ast.literal_eval(x):
        if count < 3:
            result.append(item["name"])
        count += 1
    return result

def get_director(x):
    result = []
    for item in ast.literal_eval(x):
        if item["job"] == "Director":
            result.append(item["name"])
            break
    return result

movies["genres"] = movies["genres"].apply(get_names)
movies["keywords"] = movies["keywords"].apply(get_names)
movies["cast"] = movies["cast"].apply(get_top_cast)
movies["crew"] = movies["crew"].apply(get_director)

movies["overview"] = movies["overview"].apply(lambda x: x.split())

def clean(text):
    return [i.replace(" ", "") for i in text]

movies["genres"] = movies["genres"].apply(clean)
movies["keywords"] = movies["keywords"].apply(clean)
movies["cast"] = movies["cast"].apply(clean)
movies["crew"] = movies["crew"].apply(clean)

movies["tags"] = movies["overview"] + movies["genres"] + movies["keywords"] + movies["cast"] + movies["crew"]

df = movies[["movie_id", "title", "tags"]]

df["tags"] = df["tags"].apply(lambda x: " ".join(x))
df["tags"] = df["tags"].str.lower()

vectorizer = CountVectorizer(max_features=5000, stop_words="english")
vectors = vectorizer.fit_transform(df["tags"]).toarray()

similarity = cosine_similarity(vectors)

def recommend(movie_name):
    movie_name = movie_name.lower()

    if movie_name not in df["title"].str.lower().values:
        print("Movie not found")
        return

    idx = df[df["title"].str.lower() == movie_name].index[0]
    distances = list(enumerate(similarity[idx]))
    distances = sorted(distances, key=lambda x: x[1], reverse=True)

    for i in distances[1:6]:
        print(df.iloc[i[0]].title)

recommend("Inception")
