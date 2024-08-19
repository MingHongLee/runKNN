# python -m streamlit run runKNN.py

import streamlit as st
import pandas as pd
import pickle
from fuzzywuzzy import process
import numpy as np

# Title and description of the Streamlit app
st.title('Music Recommender System')
st.write("Step 2: Enter a song name and get 5 similar song recommendations based on content similarity")

# Adding a sidebar for User ID input
st.sidebar.title("Step 1:")
user_id = st.sidebar.text_input('Please enter your User ID', '')

try:
    user_id = int(user_id)
except ValueError:
    st.sidebar.error("Please enter a valid number")

# Load your preprocessed dataset
df = pd.read_csv('testpca.csv')  # Preprocessed music data with numerical features

# Load the trained KNN model from the pickle file
with open('knn_model.pkl', 'rb') as f:
    knn10 = pickle.load(f)

# Use the relevant features for similarity calculation
X = df

# Input field for song name
song_input = st.text_input("Enter a song name:")

# Recommendation function
def recommender(song_name, recommendation_set, model):
    # Use fuzzy matching to find the closest song name in the dataset
    idx = process.extractOne(song_name, recommendation_set['name'])[2]
    st.write(f"Song Selected: {df['name'][idx]} by {df['artist'][idx]}")

    # Select the required numerical columns for recommendation
    required_songs = recommendation_set.select_dtypes(np.number).drop(columns=['cat', 'cluster', 'year']).copy()

    # Find 5 nearest neighbors using KNN
    distances, indices = model.kneighbors(required_songs.iloc[idx].values.reshape(1, -1))
    
    return idx, indices.flatten()

# Initialize session state for playlist
if 'playlist' not in st.session_state:
    st.session_state.playlist = pd.DataFrame(columns=['Song', 'Artist', 'Music Genre Tags', 'Original Song'])

# If the user has entered a song name, perform the recommendation
if song_input:
    original_idx, indices = recommender(song_input, X, knn10)

    # Prepare the data for the table
    table_data = {
        "Song": [df['name'][i] for i in indices],
        "Artist": [df['artist'][i] for i in indices],
        "Music Genre Tags": [df['tags'][i] for i in indices]
    }

    # Convert the data into a DataFrame
    table_df = pd.DataFrame(table_data)

    # Filter to show only songs 2 to 6 (index 1 to 5)
    filtered_df = table_df.iloc[1:6].reset_index(drop=True)

    # Display the filtered table with checkboxes for selection
    st.write("Step 3: You may select any recommended songs below and click on the 'Add to Playlist' button to create your personal playlist")

    # Display the filtered DataFrame with checkboxes
    selected_indices = []
    for idx, row in filtered_df.iterrows():
        if st.checkbox(f"{row.Song} by {row.Artist}", key=idx):
            selected_indices.append(idx)

    # Filter selected songs
    selected_songs = filtered_df.loc[selected_indices]

    # If the user clicks the "Add to Playlist" button, show the selected songs
    if st.button('Add to Playlist'):
        if not selected_songs.empty:
            # Add the original song to the playlist DataFrame
            original_song = pd.DataFrame([{
                'Song': df['name'][original_idx],
                'Artist': df['artist'][original_idx],
                'Music Genre Tags': df['tags'][original_idx],
                'Original Song': True
            }])
            
            st.write("Your Playlist")
            st.dataframe(selected_songs, use_container_width=True, hide_index=True)
            # Update session state playlist
            st.session_state.playlist = pd.concat([st.session_state.playlist, original_song, selected_songs], ignore_index=True)
            # Save the updated playlist to a CSV file
            if user_id:
                filename = f'Playlist_{user_id}.csv'
                st.session_state.playlist.to_csv(filename, index=False)
                st.write(f"Playlist saved as {filename}")
        else:
            st.write("No songs selected, please try again.")
