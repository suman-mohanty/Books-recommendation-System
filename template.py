import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the saved objects
model = pickle.load(open('artifacts/model.pkl', 'rb'))
book_names = pickle.load(open('artifacts/book_names.pkl', 'rb'))
final_rating = pickle.load(open('artifacts/final_rating.pkl', 'rb'))
book_pivot = pickle.load(open('artifacts/book_pivot.pkl', 'rb'))

def recommend_book(book_name):
    if book_name not in book_pivot.index:
        return ["Book not found in the dataset."]
    
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1), n_neighbors=6)
    
    recommended_books = []
    for i in range(len(suggestion)):
        books = book_pivot.index[suggestion[i]]
        for j in books:
            if j != book_name:
                recommended_books.append(j)
    
    return recommended_books

def display_recommendations(book_name):
    recommendations = recommend_book(book_name)
    
    if len(recommendations) == 1 and recommendations[0] == "Book not found in the dataset.":
        st.write(recommendations[0])
    else:
        st.write(f"Recommendations for '{book_name}':")
        cols = st.columns(len(recommendations))
        for idx, book in enumerate(recommendations):
            with cols[idx]:
                st.write(book)
                ids = np.where(final_rating['title'] == book)[0][0]
                url = final_rating.iloc[ids]['image_url']
                st.image(url, width=100)
                if st.button(f"Recommend based on '{book}'", key=book):
                    st.session_state['book_name'] = book
                    st.experimental_rerun()

if 'book_name' not in st.session_state:
    st.session_state['book_name'] = ''

st.title('Book Recommendation System')

book_name = st.text_input('Enter a book name:', value=st.session_state['book_name'])

if st.button('Recommend', key='main'):
    display_recommendations(book_name)
