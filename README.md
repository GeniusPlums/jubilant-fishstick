# Content-Based Recommendation System

This Streamlit application demonstrates a content-based recommendation system that finds similar items based on text descriptions. It allows users to upload a dataset of items with descriptions, process the text using NLP techniques, and get recommendations for similar items.

## Project Objective

The goal of this project is to build a recommendation system that:
- Processes text data using natural language processing techniques
- Converts text descriptions into numerical features
- Calculates similarity between items based on their descriptions
- Recommends similar items based on content similarity

## Features

- **File Upload**: Support for CSV and Excel files containing items and their descriptions
- **Text Processing**: Configurable text preprocessing options including:
  - Stopword removal
  - Stemming
  - Feature selection parameters
- **Vectorization Methods**: Support for TF-IDF and Count vectorization
- **Interactive UI**: Select items and see top recommendations
- **Visualization**: Bar charts showing similarity scores
- **Feature Importance**: Display of top terms that influenced recommendations

## Installation and Setup

1. Clone this repository:
   ```
   git clone <repository-url>
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   streamlit run app.py
   