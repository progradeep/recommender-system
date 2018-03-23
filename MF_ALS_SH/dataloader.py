import pandas as pd
import numpy as np

def dataloader():

    movie_id = pd.read_csv("C:\\Users\msi\Desktop\Soohyun\CHALLENGERS\DATASET\GroupLens\\movies.csv").drop(["title", "genres"], axis=1)
    movie_id["movie_index"] = movie_id.index


    rating_data = pd.read_csv("C:\\Users\msi\Desktop\Soohyun\CHALLENGERS\DATASET\GroupLens\\ratings.csv").drop(["timestamp"], axis=1)

    rating_data = rating_data.merge(movie_id, how="left", left_on="movieId", right_on="movieId")

    rp = rating_data.pivot_table(columns=["movieId"], index=["userId"],values=["rating"])

    rp = rp.fillna(0)

    return rp.values
