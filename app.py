import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")
import streamlit as st


def cluster_creation(df):
    # Columns used for clustering 
    cols_for_clustering = ["Country", 
                           "Education", 
                           "Income", 
                           "Total_Spent", 
                           "Living_With", 
                           "Total_Children",
                            "Is_Parent"]
    df_cluster = df[cols_for_clustering]
    df_cluster = pd.get_dummies(df_cluster, dtype=int)
    
    # Normalizing
    scale_norm = StandardScaler()
    df_cluster_std = pd.DataFrame(scale_norm.fit_transform(df_cluster), columns=df_cluster.columns)
    
    # Perform K-means clustering with 10 clusters
    kmeans = KMeans(n_clusters=10, random_state=0)
    df['Cluster'] = kmeans.fit_predict(df_cluster_std)
    df["Cluster"] += 1
    return df


def main():
    # Title of the app
    st.title("GenAI in Marketing")

    # Initialize session state to track file upload and segment selection
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
    if 'selected_segment' not in st.session_state:
        st.session_state.selected_segment = None

    # Define a function to handle segment selection
    def on_segment_select():
        st.session_state.selected_segment = st.session_state.segment

    # Upload file section
    if not st.session_state.file_uploaded:
        # File uploader
        uploaded_file = st.file_uploader("Upload File:", type=["csv", "xlsx", "json"], label_visibility="visible")

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)  # Assuming the file is a CSV for simplicity
            df = cluster_creation(df)
            st.session_state.df = df  # Store the DataFrame in session state
            st.session_state.file_uploaded = True
            st.experimental_rerun()
    else:
        # Creating columns for layout with width ratio
        left_column, right_column = st.columns([1, 3])

        # Left column: Segments dropdown
        with left_column:
            # Segments dropdown
            segments = ["None", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            st.selectbox("Segments", segments, key='segment', on_change=on_segment_select)

        # Right column: Characteristic Prompt and Stable Diffusion Prompt
        if st.session_state.selected_segment and st.session_state.selected_segment != "None":
            with right_column:
                cluster_number = st.session_state.selected_segment
                df = st.session_state.df
                cluster_data = df[df['Cluster'] == cluster_number]

                st.subheader(f"Characteristic Prompt for Segment {cluster_number}:")
                characteristic_prompt = f"""
                - Demographics:
                    - Age: 
                    - Income: 
                    - Location: 
                    - Family Status: 
                - Psychographics:
                    - They tend to purchase luxury and premium products, likely due to their higher income and stability.
                - Behavioral Characteristics:
                    - Spend quality on wine and meat products
                - Preferred Marketing Channels:
                    - These individuals can be approached through catalog marketing due to their preference for premium and curated selections.
                """
                st.text_area("Characteristic Prompt", characteristic_prompt, height=150)

                st.subheader(f"Stable Diffusion Prompt for Segment {cluster_number}:")
                stable_diffusion_prompt = f"""
                Married couple aged 60-80, having upper income with no children, enjoying their retirement by indulging in wine and meat for their occasional dinners. They are selecting premium items in a high-end store.
                """
                st.text_area("Stable Diffusion Prompt", stable_diffusion_prompt, height=100)

                if st.button("Generate Image"):
                    st.write("Image generation would occur here.")
        else:
            with right_column:
                st.write("Select a segment to see the characteristic and Stable Diffusion prompts.")


if __name__ == "__main__":
    main()
