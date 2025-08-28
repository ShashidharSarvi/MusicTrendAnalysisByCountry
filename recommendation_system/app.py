#streamlit run app.py 
#pip install -r requirements.txt
import streamlit as st
import pandas as pd
import numpy as np

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

@st.cache_data
def load_data():
    """Load the music dataset"""
    try:
        df = pd.read_csv("cleaned_genres_data.csv")
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

def recommend_songs(df, age, num_recommendations=10):
    """Generate song recommendations based on age"""
    group = get_age_group(age)
    prefs = age_groups[group]
    
    # Filter songs based on age group preferences
    filtered_songs = df[
        (df["track_genre"].str.lower().isin([g.lower() for g in prefs["genres"]])) &
        (df["danceability"] >= prefs["min_danceability"]) &
        (df["energy"] >= prefs["min_energy"]) &
        (df["valence"] >= prefs["min_valence"])
    ].copy()
    
    if len(filtered_songs) == 0:
        # Fallback: just filter by genre if no songs match all criteria
        filtered_songs = df[
            df["track_genre"].str.lower().isin([g.lower() for g in prefs["genres"]])
        ].copy()
    
    if len(filtered_songs) == 0:
        # Final fallback: return most popular songs
        filtered_songs = df.copy()
    
    # Sort by popularity and get top recommendations
    recommendations = filtered_songs.nlargest(num_recommendations, "popularity")
    
    return recommendations, group, prefs["description"]

def search_songs(df, query):
    """Search for songs by name or artist"""
    query = query.lower()
    results = df[
        (df["track_name"].str.lower().str.contains(query, na=False)) |
        (df["artists"].str.lower().str.contains(query, na=False))
    ]
    return results.head(20)

def main():
    st.title("🎵 Age-Based Music Recommendation System")
    st.markdown("---")
    
    # Load dataset
    df = load_data()
    if df is None:
        return
    
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
            max_value=20,
            value=10
        )
        
        st.markdown("---")
        
        # Dataset info
        st.subheader("📊 Dataset Info")
        st.write(f"**Total Songs:** {len(df):,}")
        st.write(f"**Genres:** {df['track_genre'].nunique()}")
        st.write(f"**Average Popularity:** {df['popularity'].mean():.1f}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Get recommendations button
        if st.button("🎯 Get My Recommendations", type="primary"):
            with st.spinner("Finding perfect songs for you..."):
                recommendations, age_group, description = recommend_songs(df, age, num_recs)
            
            st.success(f"✅ Found {len(recommendations)} songs for **{age_group}** ({age} years old)")
            st.info(f"**Profile:** {description}")
            
            # Display recommendations
            st.subheader("🎵 Your Recommended Songs")
            
            for idx, (_, song) in enumerate(recommendations.iterrows(), 1):
                with st.container():
                    col_a, col_b, col_c = st.columns([0.5, 3, 1])
                    
                    with col_a:
                        st.write(f"**{idx}.**")
                    
                    with col_b:
                        st.write(f"**{song['track_name']}**")
                        st.write(f"by *{song['artists']}* • {song['track_genre'].title()}")
                    
                    with col_c:
                        st.metric("Popularity", f"{song['popularity']}")
                    
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
        st.subheader("🔍 Search Songs")
        search_query = st.text_input("Search by song title or artist:")
        
        if search_query:
            search_results = search_songs(df, search_query)
            st.write(f"Found {len(search_results)} results:")
            
            for _, song in search_results.head(5).iterrows():
                st.write(f"🎵 **{song['track_name']}**")
                st.write(f"*{song['artists']}* • Popularity: {song['popularity']}")
                st.write("---")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🎵 Built with Streamlit | Music Recommendation System</p>
        <p><small>Recommendations based on age group listening trends and song characteristics</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()