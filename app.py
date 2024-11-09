import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the saved objects
model = pickle.load(open('artifacts/model.pkl', 'rb'))
book_names = pickle.load(open('artifacts/book_names.pkl', 'rb'))
final_rating = pickle.load(open('artifacts/final_rating.pkl', 'rb'))
book_pivot = pickle.load(open('artifacts/book_pivot.pkl', 'rb'))

# Add custom CSS for styling
st.markdown("""
    <style>
    body{
        background-color:#ffffff;
    }
    .title {
        font-size: 50px;
        color: #4CAF50;
        text-align: center;
    }
    .recommendation {
        font-size: 20px;
        color: #FF6347;
        text-align: center;
    }
    .book-title {
        font-size: 18px;
        color: #FFFFFF;
        text-align: center;
    }
    .button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
    }
    </style>
    """, unsafe_allow_html=True)

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
        st.markdown(f"<div class='recommendation'>Recommendations for '{book_name}':</div>", unsafe_allow_html=True)
        cols = st.columns(len(recommendations))
        for idx, book in enumerate(recommendations):
            with cols[idx]:
                st.markdown(f"<div class='book-title'>{book}</div>", unsafe_allow_html=True)
                ids = np.where(final_rating['title'] == book)[0][0]
                url = final_rating.iloc[ids]['image_url']
                st.image(url, width=100)
                #if st.button(f"Recommend based on '{book}'", key=f"recommend_{book}"):
                    #st.session_state['book_name'] = final_rating.iloc[ids][book]
                    #st.experimental_rerun()

if 'book_name' not in st.session_state:
    st.session_state['book_name'] = ''

st.markdown("<div class='title'>Book Recommendation System</div>", unsafe_allow_html=True)

book_name = st.text_input('Enter a book name:', value=st.session_state['book_name'])

if st.button('Recommend', key='main'):
    display_recommendations(book_name)

# Display recommendations if a book name is in session state
if st.session_state['book_name']:
    display_recommendations(st.session_state['book_name'])
