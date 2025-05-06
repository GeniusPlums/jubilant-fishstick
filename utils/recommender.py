import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp

class RecommendationEngine:
    """
    A class for generating content-based recommendations based on text similarity.
    """
    
    def __init__(self, items, feature_matrix, vectorizer=None):
        """
        Initialize the RecommendationEngine with items and their feature matrix.
        
        Args:
            items (list): List of item identifiers
            feature_matrix (numpy.ndarray): Feature matrix of items
            vectorizer (object, optional): The vectorizer used to create the feature matrix
        """
        self.items = items
        self.feature_matrix = feature_matrix
        self.vectorizer = vectorizer
        
        # Calculate similarity matrix
        self.similarity_matrix = cosine_similarity(feature_matrix)
        
        # Create a mapping from item to index
        self.item_to_index = {item: idx for idx, item in enumerate(items)}
    
    def get_recommendations(self, item, top_n=5, threshold=0.0):
        """
        Get recommendations for a specific item.
        
        Args:
            item (str): Item identifier
            top_n (int): Number of recommendations to return
            threshold (float): Minimum similarity threshold
            
        Returns:
            pandas.DataFrame: Dataframe with recommended items and similarity scores
        """
        # Get the index of the item
        if item not in self.item_to_index:
            raise ValueError(f"Item '{item}' not found in the dataset.")
            
        item_index = self.item_to_index[item]
        
        # Get similarity scores for this item with all other items
        similarity_scores = self.similarity_matrix[item_index]
        
        # Create a list of (index, similarity) tuples for all other items
        item_scores = [(i, score) for i, score in enumerate(similarity_scores) 
                        if i != item_index and score >= threshold]
        
        # Sort by similarity (descending)
        item_scores = sorted(item_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N
        top_items = item_scores[:top_n]
        
        # Create a DataFrame with results
        recommendations = pd.DataFrame([
            {"item": self.items[idx], "similarity": score}
            for idx, score in top_items
        ])
        
        return recommendations
    
    def get_items(self):
        """
        Get the list of items.
        
        Returns:
            list: List of item identifiers
        """
        return self.items
    
    def get_top_terms(self, item, top_n=10):
        """
        Get the top terms (features) that define an item.
        
        Args:
            item (str): Item identifier
            top_n (int): Number of top terms to return
            
        Returns:
            dict: Dictionary of top terms and their weights
        """
        # Check if we have a vectorizer
        if self.vectorizer is None:
            return None
            
        try:
            # Get the index of the item
            item_index = self.item_to_index[item]
            
            # Get the feature vector for this item
            feature_vector = self.feature_matrix[item_index]
            
            # Convert to array if sparse
            if sp.issparse(feature_vector):
                feature_vector = feature_vector.toarray().flatten()
            
            # Get feature names
            try:
                feature_names = self.vectorizer.get_feature_names_out()
            except AttributeError:
                # For older scikit-learn versions
                feature_names = self.vectorizer.get_feature_names()
            
            # Create a list of (term, weight) tuples
            term_weights = [(name, weight) for name, weight in zip(feature_names, feature_vector)]
            
            # Sort by weight (descending)
            term_weights = sorted(term_weights, key=lambda x: x[1], reverse=True)
            
            # Get top N
            top_terms = term_weights[:top_n]
            
            # Create a dictionary with results
            return {term: float(weight) for term, weight in top_terms if weight > 0}
            
        except Exception as e:
            print(f"Error getting top terms: {str(e)}")
            return None
