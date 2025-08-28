import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(
    page_title="🎵 Music Recommender",
    page_icon="🎵",
    layout="wide"
)

# Age group preferences mapping
age_groups = {
    "Teen": {
        "genres": ["pop", "hip-hop", "edm", "dance", "electronic"],
        "min_danceability": 0.6,
        "min_energy": 0.7,
        "min_valence": 0.5,
        "description": "High energy, danceable pop and hip-hop hits"
    },
    "Young Adult": {
        "genres": ["pop", "indie", "rock", "alternative", "hip-hop"],
        "min_danceability": 0.5,
        "min_energy": 0.6,
        "min_valence": 0.4,
        "description": "Mix of popular and indie tracks with good energy"
    },
    "Adult": {
        "genres": ["rock", "jazz", "acoustic", "blues", "folk", "country"],
        "min_danceability": 0.4,
        "min_energy": 0.5,
        "min_valence": 0.3,
        "description": "More mature sounds with acoustic and rock elements"
    },
    "Senior": {
        "genres": ["classical", "jazz", "acoustic", "blues", "folk", "oldies"],
        "min_danceability": 0.3,
        "min_energy": 0.4,
        "min_valence": 0.3,
        "description": "Timeless classics and acoustic sounds"
    }
}

# Audio features for similarity calculation
AUDIO_FEATURES = ['danceability', 'energy', 'loudness', 'speechiness', 
                  'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

@st.cache_data
def load_data():
    """Load the music dataset with memory optimization"""
    try:
        # Load only essential columns to save memory
        essential_cols = ['track_name', 'artists', 'album_name', 'track_genre', 
                         'popularity', 'duration_ms', 'explicit'] + AUDIO_FEATURES
        
        df = pd.read_csv("cleaned_genres_data.csv", usecols=essential_cols)
        
        # Sample dataset if too large (for testing)
        if len(df) > 50000:
            st.warning(f"⚠️ Dataset is large ({len(df):,} songs). Using a sample of 50,000 songs for better performance.")
            df = df.sample(n=50000, random_state=42).reset_index(drop=True)
        
        # Convert to more memory-efficient data types
        df['popularity'] = df['popularity'].astype('int16')
        df['duration_ms'] = df['duration_ms'].astype('int32')
        df['explicit'] = df['explicit'].astype('bool')
        
        # Fill missing values
        for col in AUDIO_FEATURES:
            df[col] = df[col].fillna(df[col].median())
        
        return df
        
    except FileNotFoundError:
        st.error("❌ Dataset file 'cleaned_genres_data.csv' not found!")
        st.info("Please make sure the dataset file is in the same directory as this script.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        return None

def get_age_group(age):
    """Map age to age group"""
    if age < 20:
        return "Teen"
    elif age < 30:
        return "Young Adult"
    elif age < 50:
        return "Adult"
    else:
        return "Senior"

def calculate_similarity_on_demand(df, selected_song_idx, candidate_indices):
    """Calculate similarity only for candidate songs to save memory"""
    
    # Get features for selected song
    selected_features = df.iloc[selected_song_idx][AUDIO_FEATURES].values.reshape(1, -1)
    
    # Get features for candidate songs
    candidate_features = df.iloc[candidate_indices][AUDIO_FEATURES].values
    
    # Standardize features
    scaler = StandardScaler()
    all_features = np.vstack([selected_features, candidate_features])
    all_features_scaled = scaler.fit_transform(all_features)
    
    # Calculate similarities (first row is selected song, rest are candidates)
    similarities = cosine_similarity(all_features_scaled[0:1], all_features_scaled[1:])
    
    return similarities[0]  # Return similarities as 1D array

def search_songs(df, query):
    """Search for songs by name or artist"""
    if not query:
        return pd.DataFrame()
    
    query = query.lower()
    results = df[
        (df["track_name"].str.lower().str.contains(query, na=False)) |
        (df["artists"].str.lower().str.contains(query, na=False))
    ]
    return results.head(20)

def get_similar_songs_optimized(df, selected_song_idx, age, num_recommendations=10):
    """Get similar songs with memory-optimized approach"""
    
    # Get age group preferences
    age_group = get_age_group(age)
    age_prefs = age_groups[age_group]
    
    # First, filter by age preferences to reduce candidate pool
    age_filtered = df[
        (df["track_genre"].str.lower().isin([g.lower() for g in age_prefs["genres"]])) |
        (df["danceability"] >= age_prefs["min_danceability"] - 0.3) |
        (df["energy"] >= age_prefs["min_energy"] - 0.3)
    ]
    
    # If still too many candidates, take top by popularity
    if len(age_filtered) > 5000:
        age_filtered = age_filtered.nlargest(5000, 'popularity')
    
    # If too few candidates, expand the search
    if len(age_filtered) < 100:
        age_filtered = df.nlargest(2000, 'popularity')
    
    # Remove the selected song from candidates
    candidate_indices = age_filtered.index[age_filtered.index != selected_song_idx].tolist()
    
    if len(candidate_indices) == 0:
        return df.sample(num_recommendations), age_group, age_prefs["description"]
    
    # Calculate similarities only for candidates
    similarities = calculate_similarity_on_demand(df, selected_song_idx, candidate_indices)
    
    # Create results dataframe
    candidates_df = df.iloc[candidate_indices].copy()
    candidates_df['similarity_score'] = similarities
    
    # Combine similarity and popularity for ranking
    max_popularity = candidates_df['popularity'].max()
    candidates_df['combined_score'] = (
        0.7 * candidates_df['similarity_score'] + 
        0.3 * (candidates_df['popularity'] / max_popularity)
    )
    
    # Get top recommendations
    recommendations = candidates_df.nlargest(num_recommendations, 'combined_score')
    
    return recommendations, age_group, age_prefs["description"]

def main():
    st.title("🎵 Personalized Music Recommendation System")
    st.markdown("### *First choose a song you like, then get similar recommendations based on your age!*")
    st.markdown("---")
    
    # Load dataset
    df = load_data()
    if df is None:
        return
    
    # Show memory usage info
    memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
    st.info(f"📊 Loaded {len(df):,} songs ({memory_usage:.1f} MB in memory)")
    
    # Initialize session state
    if 'selected_song' not in st.session_state:
        st.session_state.selected_song = None
    if 'selected_song_idx' not in st.session_state:
        st.session_state.selected_song_idx = None
    
    # Sidebar for user input
    with st.sidebar:
        st.header("🎯 Your Preferences")
        
        # Age input
        age = st.number_input(
            "Enter your age:",
            min_value=10,
            max_value=100,
            value=25,
            step=1
        )
        
        # Number of recommendations
        num_recs = st.slider(
            "Number of recommendations:",
            min_value=5,
            max_value=15,
            value=8
        )
        
        st.markdown("---")
        
        # Currently selected song
        if st.session_state.selected_song is not None:
            st.subheader("🎵 Selected Song")
            song = st.session_state.selected_song
            st.write(f"**{song['track_name']}**")
            st.write(f"*{song['artists']}*")
            st.write(f"Genre: {song['track_genre'].title()}")
            
            if st.button("🗑️ Clear Selection"):
                st.session_state.selected_song = None
                st.session_state.selected_song_idx = None
                st.rerun()
        
        st.markdown("---")
        
        # Dataset info
        st.subheader("📊 Dataset Info")
        st.write(f"**Total Songs:** {len(df):,}")
        st.write(f"**Genres:** {df['track_genre'].nunique()}")
        st.write(f"**Avg Popularity:** {df['popularity'].mean():.1f}")
        st.write(f"**Memory Usage:** {memory_usage:.1f} MB")
    
    # Main content area
    if st.session_state.selected_song is None:
        # Step 1: Song Selection
        st.subheader("🔍 Step 1: Search and Select a Song You Like")
        
        search_query = st.text_input(
            "Search by song title or artist name:",
            placeholder="e.g., 'Shape of You', 'Taylor Swift', 'Bohemian Rhapsody'"
        )
        
        if search_query:
            search_results = search_songs(df, search_query)
            
            if len(search_results) > 0:
                st.write(f"Found {len(search_results)} songs matching '{search_query}':")
                
                # Display search results
                for idx, (df_idx, song) in enumerate(search_results.iterrows()):
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.write(f"🎵 **{song['track_name']}**")
                        st.write(f"*{song['artists']}* • {song['track_genre'].title()}")
                    
                    with col2:
                        st.write(f"Popularity: {song['popularity']}")
                    
                    with col3:
                        if st.button(f"Select", key=f"select_{df_idx}"):
                            st.session_state.selected_song = song
                            st.session_state.selected_song_idx = df_idx
                            st.success(f"✅ Selected: {song['track_name']}")
                            st.rerun()
                    
                    st.markdown("---")
            else:
                st.info("No songs found. Try a different search term.")
        
        # Popular songs suggestion
        st.subheader("🔥 Or Choose from Popular Songs")
        popular_songs = df.nlargest(10, 'popularity')
        
        for idx, (df_idx, song) in enumerate(popular_songs.iterrows()):
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"🎵 **{song['track_name']}**")
                st.write(f"*{song['artists']}* • {song['track_genre'].title()}")
            
            with col2:
                st.write(f"Popularity: {song['popularity']}")
            
            with col3:
                if st.button(f"Select", key=f"popular_{df_idx}"):
                    st.session_state.selected_song = song
                    st.session_state.selected_song_idx = df_idx
                    st.success(f"✅ Selected: {song['track_name']}")
                    st.rerun()
            
            st.markdown("---")
    
    else:
        # Step 2: Show Recommendations
        st.subheader("🎯 Step 2: Your Personalized Recommendations")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("🎵 Get Similar Songs Based on My Selection!", type="primary", use_container_width=True):
                with st.spinner("Finding songs similar to your selection..."):
                    try:
                        recommendations, age_group, description = get_similar_songs_optimized(
                            df, st.session_state.selected_song_idx, age, num_recs
                        )
                    except Exception as e:
                        st.error(f"Error generating recommendations: {str(e)}")
                        return
                
                st.success(f"✅ Found {len(recommendations)} similar songs for **{age_group}** ({age} years old)")
                st.info(f"**Based on:** '{st.session_state.selected_song['track_name']}' by {st.session_state.selected_song['artists']}")
                st.info(f"**Your Profile:** {description}")
                
                # Display recommendations
                st.subheader("🎵 Songs Similar to Your Selection")
                
                for idx, (_, song) in enumerate(recommendations.iterrows(), 1):
                    with st.container():
                        col_a, col_b, col_c, col_d = st.columns([0.5, 3, 1, 1])
                        
                        with col_a:
                            st.write(f"**{idx}.**")
                        
                        with col_b:
                            st.write(f"**{song['track_name']}**")
                            st.write(f"by *{song['artists']}* • {song['track_genre'].title()}")
                        
                        with col_c:
                            st.metric("Popularity", f"{song['popularity']}")
                        
                        with col_d:
                            if 'similarity_score' in song:
                                st.metric("Similarity", f"{song['similarity_score']:.2f}")
                        
                        # Additional details in expander
                        with st.expander(f"Details for '{song['track_name']}'"):
                            details_col1, details_col2 = st.columns(2)
                            
                            with details_col1:
                                st.write(f"**Album:** {song['album_name']}")
                                st.write(f"**Duration:** {song['duration_ms'] // 60000}:{(song['duration_ms'] % 60000) // 1000:02d}")
                                st.write(f"**Explicit:** {'Yes' if song['explicit'] else 'No'}")
                            
                            with details_col2:
                                st.write(f"**Danceability:** {song['danceability']:.2f}")
                                st.write(f"**Energy:** {song['energy']:.2f}")
                                st.write(f"**Valence:** {song['valence']:.2f}")
                        
                        st.markdown("---")
        
        with col2:
            if st.button("🔄 Choose Different Song", use_container_width=True):
                st.session_state.selected_song = None
                st.session_state.selected_song_idx = None
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🎵 Built with Streamlit | Memory-Optimized Music Recommender</p>
        <p><small>Recommendations based on song similarity and age group preferences</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()