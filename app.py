import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from utils.text_processor import TextProcessor
from utils.recommender import RecommendationEngine

def main():
    st.set_page_config(
        page_title="Content-Based Recommendation System",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("Content-Based Recommendation System")
    st.markdown("""
    This app demonstrates a content-based recommendation system that finds similar items 
    based on their text descriptions. Upload a dataset with items and their descriptions 
    to get started.
    """)
    
    # Session state initialization
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'recommender' not in st.session_state:
        st.session_state.recommender = None
    if 'text_processor' not in st.session_state:
        st.session_state.text_processor = None
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    
    # Data Upload Section
    st.header("1. Upload Your Data")
    st.markdown("""
    Upload a CSV or Excel file with your items. The file should have at least two columns:
    - A column with item identifiers (e.g., names, titles, IDs)
    - A column with text descriptions
    """)
    
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            # Determine file type and read accordingly
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
            # Check if data has at least 2 columns
            if data.shape[1] < 2:
                st.error("Your file must have at least 2 columns: one for item identifiers and one for descriptions.")
                return
            
            # Display the data preview
            st.subheader("Data Preview")
            st.dataframe(data.head(5))
            
            # Store the data in session state
            st.session_state.data = data
            
            # Column Selection
            st.subheader("2. Select Columns")
            
            id_col = st.selectbox(
                "Select the column containing item identifiers:",
                options=data.columns.tolist()
            )
            
            desc_col = st.selectbox(
                "Select the column containing item descriptions:",
                options=[col for col in data.columns if col != id_col],
                index=0 if id_col != data.columns[1] else 1
            )
            
            # Text Processing Parameters
            st.subheader("3. Text Processing Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                remove_stopwords = st.checkbox("Remove stopwords", value=True)
                stemming = st.checkbox("Apply stemming", value=False)
            
            with col2:
                min_df = st.slider("Minimum document frequency (%)", 0, 100, 5) / 100
                max_features = st.slider("Maximum number of features", 100, 10000, 5000, step=100)
            
            vectorizer_type = st.radio(
                "Select vectorization method:",
                options=["TF-IDF", "Count Vectorizer"],
                horizontal=True
            )
            
            # Process Button
            process_btn = st.button("Process Data")
            
            if process_btn:
                with st.spinner("Processing text data..."):
                    try:
                        # Create text processor instance
                        text_processor = TextProcessor(
                            remove_stopwords=remove_stopwords,
                            apply_stemming=stemming,
                            min_df=min_df,
                            max_features=max_features,
                            vectorizer_type=vectorizer_type
                        )
                        
                        # Process the text
                        items = data[id_col].tolist()
                        descriptions = data[desc_col].tolist()
                        
                        # Check for missing values
                        if data[desc_col].isnull().sum() > 0:
                            st.warning(f"Found {data[desc_col].isnull().sum()} missing descriptions. These items will be filtered out.")
                            valid_indices = ~data[desc_col].isnull()
                            items = data.loc[valid_indices, id_col].tolist()
                            descriptions = data.loc[valid_indices, desc_col].tolist()
                        
                        # Check if we have any valid items left
                        if len(items) < 2:
                            st.error("Not enough valid items with descriptions to proceed. Please check your data.")
                            return
                            
                        # Process the data
                        text_processor.fit_transform(descriptions)
                        
                        # Create recommender
                        recommender = RecommendationEngine(
                            items=items,
                            feature_matrix=text_processor.get_feature_matrix(),
                            vectorizer=text_processor.get_vectorizer()
                        )
                        
                        # Store in session state
                        st.session_state.text_processor = text_processor
                        st.session_state.recommender = recommender
                        st.session_state.processed = True
                        st.session_state.id_col = id_col
                        st.session_state.desc_col = desc_col
                        
                        st.success("Text processing complete! You can now get recommendations.")
                        st.rerun()
                    
                    except Exception as e:
                        st.error(f"An error occurred during processing: {str(e)}")
            
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    # Recommendation section
    if st.session_state.processed and st.session_state.recommender:
        st.header("4. Get Recommendations")
        
        # Item selection
        selected_item = st.selectbox(
            "Select an item to find similar items:",
            options=st.session_state.recommender.get_items()
        )
        
        num_recommendations = st.slider(
            "Number of recommendations:",
            min_value=1,
            max_value=min(20, len(st.session_state.recommender.get_items()) - 1),
            value=5
        )
        
        similarity_threshold = st.slider(
            "Minimum similarity threshold:",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05
        )
        
        if st.button("Get Recommendations"):
            with st.spinner("Finding similar items..."):
                try:
                    # Get recommendations
                    recommendations = st.session_state.recommender.get_recommendations(
                        item=selected_item,
                        top_n=num_recommendations,
                        threshold=similarity_threshold
                    )
                    
                    if recommendations.empty or len(recommendations) == 0:
                        st.info("No recommendations found above the similarity threshold. Try lowering the threshold.")
                    else:
                        st.subheader("Recommended Items")
                        
                        # Display the selected item details
                        selected_item_data = st.session_state.data[
                            st.session_state.data[st.session_state.id_col] == selected_item
                        ]
                        
                        st.write("**Selected Item:**")
                        st.info(f"**{selected_item}**\n\n{selected_item_data[st.session_state.desc_col].values[0]}")
                        
                        # Display recommendations
                        st.write("**Recommendations:**")
                        
                        for _, row in recommendations.iterrows():
                            item = row['item']
                            similarity = row['similarity']
                            
                            item_data = st.session_state.data[
                                st.session_state.data[st.session_state.id_col] == item
                            ]
                            
                            description = item_data[st.session_state.desc_col].values[0]
                            
                            with st.expander(f"{item} (Similarity: {similarity:.4f})"):
                                st.write(description)
                        
                        # Display similarity scores in a bar chart
                        st.subheader("Similarity Scores")
                        
                        chart_data = recommendations.copy()
                        chart_data.set_index('item', inplace=True)
                        st.bar_chart(chart_data)
                        
                        # Show important features
                        st.subheader("Important Features")
                        st.markdown("""
                        The following terms had the most influence on these recommendations:
                        """)
                        
                        top_terms = st.session_state.recommender.get_top_terms(
                            item=selected_item,
                            top_n=10
                        )
                        
                        if top_terms is not None:
                            st.write(top_terms)
                        else:
                            st.info("Feature importance extraction not available for this vectorizer.")
                
                except Exception as e:
                    st.error(f"An error occurred while generating recommendations: {str(e)}")
    
    # Add information about the app
    st.sidebar.title("About")
    st.sidebar.info(
        "This application demonstrates a content-based recommendation system "
        "using text similarity. Upload a dataset with items and descriptions, "
        "then get recommendations for similar items."
    )
    
    st.sidebar.title("How It Works")
    st.sidebar.markdown(
        """
        1. **Upload Data**: CSV or Excel file with items and descriptions
        2. **Select Columns**: Identify which columns contain item identifiers and descriptions
        3. **Configure Settings**: Adjust text processing parameters
        4. **Get Recommendations**: Select an item to see similar items
        
        The system processes text using natural language processing techniques, 
        converts descriptions into numerical features, and calculates similarities 
        between items based on their descriptions.
        """
    )

if __name__ == "__main__":
    main()
