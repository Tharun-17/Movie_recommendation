import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import streamlit as st

movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

final_dataset = ratings.pivot(index='movieId',columns='userId',values='rating')

final_dataset.fillna(0,inplace=True)

no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
no_movies_voted = ratings.groupby('userId')['rating'].agg('count')

final_dataset = final_dataset.loc[no_user_voted[no_user_voted > 10].index,:]

final_dataset=final_dataset.loc[:,no_movies_voted[no_movies_voted > 50].index]


csr_data = csr_matrix(final_dataset.values)
final_dataset.reset_index(inplace=True)

knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)

movies['title'] = [i.lower() for i in movies['title']]

avg_ratings = ratings.groupby('movieId').mean()

movies_with_ratings = pd.merge(movies, avg_ratings, on='movieId', how='left')

movies = movies_with_ratings

def get_movie_recommendation(movie_name):
    movie_name = movie_name.lower()

    n_movies_to_reccomend = 10
    movie_list = movies[movies['title'].str.contains(movie_name)]  
    if len(movie_list):        
        movie_idx = movie_list.iloc[0]['movieId']
        movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]
        final_movie = movies[movies['movieId']==movie_idx]
        distances , indices = knn.kneighbors(csr_data[movie_idx],n_neighbors=n_movies_to_reccomend+1)    
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
        recommend_frame = []
        for val in rec_movie_indices:
            movie_idx = final_dataset.iloc[val[0]]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            recommend_frame.append({'Title':movies.iloc[idx]['title'].values[0],'Distance':val[1],'Genres':movies.iloc[idx]['genres'].values[0]})
        final_movie = movies[movies['movieId']==movie_idx]
        df = pd.DataFrame(recommend_frame,index=range(1,n_movies_to_reccomend+1))
        recommendation_rating = df
        
        genres_in = final_movie['genres'].values
        genres_in = genres_in[0].split('|')
        recommended_df = pd.DataFrame()
        for genre in genres_in:
            rec_df = movies[movies['genres'].str.contains(genre)].sort_values(by = 'rating', ascending = False).head(2)
            recommended_df = pd.concat([recommended_df,rec_df])
        recommendation_genre = recommended_df[['title','genres','rating']]
        
        return recommendation_rating, recommendation_genre
    else:
        return "No movies found. Please check your input"
    
st.header('Movie Recommender.....')

movie_name = st.text_input('Give me a movie name to recommend similar movies')

choice = st.radio('Select on what basis you need recommendation',['Ratings','Genres'])

if choice == 'Ratings':
    recommendations = get_movie_recommendation(movie_name)[0]
else:
    recommendations = get_movie_recommendation(movie_name)[1]

if st.button('Get recommendations'):
    st.dataframe(recommendations)