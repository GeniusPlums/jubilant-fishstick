import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

class TextProcessor:
    """
    A class for processing text data and converting it to numerical features.
    """
    
    def __init__(self, remove_stopwords=True, apply_stemming=False, min_df=0.05, 
                 max_features=5000, vectorizer_type="TF-IDF"):
        """
        Initialize the TextProcessor with the specified parameters.
        
        Args:
            remove_stopwords (bool): Whether to remove stopwords
            apply_stemming (bool): Whether to apply stemming
            min_df (float): Minimum document frequency for feature selection
            max_features (int): Maximum number of features to extract
            vectorizer_type (str): Type of vectorizer to use ('TF-IDF' or 'Count Vectorizer')
        """
        self.remove_stopwords = remove_stopwords
        self.apply_stemming = apply_stemming
        self.min_df = min_df
        self.max_features = max_features
        self.vectorizer_type = vectorizer_type
        
        # Download NLTK resources if needed
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        # Initialize stemmer
        self.stemmer = PorterStemmer() if apply_stemming else None
        
        # Initialize stopwords
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else None
        
        # Initialize vectorizer
        if vectorizer_type == "TF-IDF":
            self.vectorizer = TfidfVectorizer(
                min_df=min_df,
                max_features=max_features,
                stop_words='english' if remove_stopwords else None,
                ngram_range=(1, 2)
            )
        else:  # Count Vectorizer
            self.vectorizer = CountVectorizer(
                min_df=min_df,
                max_features=max_features,
                stop_words='english' if remove_stopwords else None,
                ngram_range=(1, 2)
            )
        
        self.feature_matrix = None
        self.processed_text = None
    
    def preprocess_text(self, text_list):
        """
        Preprocess a list of text documents.
        
        Args:
            text_list (list): List of text documents
            
        Returns:
            list: List of preprocessed text documents
        """
        processed_texts = []
        
        for text in text_list:
            if pd.isna(text):
                processed_texts.append("")
                continue
                
            # Convert to string if not already
            if not isinstance(text, str):
                text = str(text)
            
            # Convert to lowercase
            text = text.lower()
            
            # Remove special characters and digits
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\d+', ' ', text)
            
            # Tokenize
            tokens = text.split()
            
            # Remove stopwords
            if self.remove_stopwords:
                tokens = [token for token in tokens if token not in self.stop_words]
            
            # Apply stemming
            if self.apply_stemming:
                tokens = [self.stemmer.stem(token) for token in tokens]
            
            # Join tokens back into a string
            processed_text = ' '.join(tokens)
            
            processed_texts.append(processed_text)
        
        return processed_texts
    
    def fit_transform(self, text_list):
        """
        Preprocess text and fit the vectorizer.
        
        Args:
            text_list (list): List of text documents
            
        Returns:
            numpy.ndarray: Feature matrix
        """
        self.processed_text = self.preprocess_text(text_list)
        self.feature_matrix = self.vectorizer.fit_transform(self.processed_text)
        return self.feature_matrix
    
    def transform(self, text_list):
        """
        Preprocess text and transform using the fitted vectorizer.
        
        Args:
            text_list (list): List of text documents
            
        Returns:
            numpy.ndarray: Feature matrix
        """
        processed_text = self.preprocess_text(text_list)
        return self.vectorizer.transform(processed_text)
    
    def get_feature_matrix(self):
        """
        Get the feature matrix.
        
        Returns:
            numpy.ndarray: Feature matrix
        """
        return self.feature_matrix
    
    def get_feature_names(self):
        """
        Get the feature names.
        
        Returns:
            list: Feature names
        """
        try:
            return self.vectorizer.get_feature_names_out()
        except AttributeError:
            # For older scikit-learn versions
            return self.vectorizer.get_feature_names()
    
    def get_vectorizer(self):
        """
        Get the vectorizer.
        
        Returns:
            object: Vectorizer instance
        """
        return self.vectorizer
