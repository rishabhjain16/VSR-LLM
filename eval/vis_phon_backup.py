#!/usr/bin/env python3
import os
import json
import numpy as np
import math
from collections import Counter, defaultdict
from itertools import chain
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Patch
from datetime import datetime
import panphon
import panphon.distance
import panphon.featuretable
from difflib import SequenceMatcher
import nltk
from nltk.translate import meteor_score
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import torch
from sentence_transformers import SentenceTransformer
import bert_score
import csv
import re

class LipReadingEvaluator:
    """Comprehensive lip reading evaluation system with phonetic feature weighting"""
    
    def __init__(self):
        """Initialize the evaluator with phoneme mappings and converters"""
        
        try:
            # Import phonemizer and set up the backend
            from phonemizer import phonemize
            from phonemizer.separator import Separator
            from phonemizer.backend import EspeakBackend
            
            # Create backend instance
            self.phonemize = phonemize
            self.separator = Separator(word=' ', phone='')
            self.phonemizer_backend = EspeakBackend('en-us', with_stress=False)
        except ImportError:
            print("ERROR: phonemizer library is required for IPA conversion")
            print("Install with: pip install phonemizer")
            raise
        
        try:
            import panphon
            self.ft = panphon.featuretable.FeatureTable()
            self.dst = panphon.distance.Distance()
        except ImportError:
            print("ERROR: panphon library is required for phonetic features")
            print("Install with: pip install panphon")
            raise
        
        # Update regex pattern for text normalization
        import re
        self.alphanumeric_pattern = re.compile(r'[^a-zA-Z0-9\s]')
        
        # Define IPA phoneme-to-viseme mapping
        self.phoneme_to_viseme = {
            # 0: Silence (only empty string and space needed now that punctuation is handled separately)
            '': 0, ' ': 0,
            
            # 1: Open/central vowels (æ, ə, ʌ)
            'æ': 1, 'ə': 1, 'ʌ': 1, 'a': 1, 'ɐ': 1,
            'æː': 1, 'æ̃': 1,  # Add long and nasalized variants
            
            # 2: Open back vowels
            'ɑ': 2, 'ɒ': 2, 'ɑː': 2, 'ɑ̃': 2, 'ɒ̃': 2,  # Add nasalized variants
            
            # 3: Open-mid back rounded
            'ɔ': 3, 'ɔː': 3, 'ɔ̃': 3,  # Add nasalized variant
            
            # 4: Mid vowels
            'ɛ': 4, 'ʊ': 4, 'e': 4, 'ɜ': 4, 'ɜː': 4, 'ɛː': 4, 'eː': 4,
            'ɛ̃': 4, 'ẽ': 4, 'ʊ̃': 4,  # Add nasalized variants
            'ə̃': 4,  # Nasalized schwa
            
            # 5: R-colored vowels
            'ɝ': 5, 'ɚ': 5, 'ɹ̩': 5, 'ɻ': 5, 'r̩': 5,  # Add syllabic r
            'ɝː': 5, 'ɚː': 5,  # Long r-colored vowels
            
            # 6: Close front vowels + /j/
            'i': 6, 'ɪ': 6, 'j': 6, 'iː': 6, 'ɪː': 6, 'y': 6,
            'eɪ': 6, 'ej': 6, 'ɪə': 6, 'iə': 6,
            'ĩ': 6, 'ɪ̃': 6,  # Add nasalized variants
            'ɨ': 6,  # Add barred i (sometimes used in unstressed positions)
            
            # 7: Close back rounded + /w/
            'u': 7, 'w': 7, 'uː': 7, 'ʍ': 7,
            'ũ': 7, 'w̥': 7,  # Add nasalized u and voiceless w
            
            # 8: Close-mid back rounded
            'o': 8, 'oː': 8, 'oʊ': 8, 'ow': 8, 'õ': 8,  # Add nasalized o
            'əʊ': 8,  # British English variant of /oʊ/
            
            # 9-11: Major diphthongs
            'aʊ': 9, 'aw': 9, 'aːʊ': 9, 'ãʊ̃': 9,  # Add long and nasalized variants
            'ɔɪ': 10, 'oj': 10, 'ɔːɪ': 10, 'ɔ̃ɪ̃': 10,  # Add long and nasalized variants
            'aɪ': 11, 'aj': 11, 'aːɪ': 11, 'ãɪ̃': 11,  # Add long and nasalized variants
            'eə': 4, 'ʊə': 4, 'ɛə': 4, 'ɪə': 4,  # Various centring diphthongs
            
            # 12: Glottal
            'h': 12, 'ʔ': 12, 'ɦ': 12, 'h̃': 12,  # Add breathy/voiced h
            
            # 13: Rhotic approximant
            'ɹ': 13, 'r': 13, 'ɾ': 13, 'ɻ': 13, 'ʀ': 13, 'ʁ': 13,
            'ɹ̥': 13, 'ɹ̩': 13, 'rː': 13,  # Add syllabic and voiceless variants
            
            # 14: Lateral approximant
            'l': 14, 'ɬ': 14, 'ɭ': 14, 'ɫ': 14, 'ʎ': 14, 'l̩': 14,
            'l̥': 14, 'lː': 14,  # Add voiceless and long variants
            
            # 15: Alveolar fricatives
            's': 15, 'z': 15, 'sː': 15, 'zː': 15,  # Add long variants
            
            # 16: Post-alveolar sounds
            'ʃ': 16, 'ʒ': 16, 'tʃ': 16, 'dʒ': 16, 'ʂ': 16, 'ʐ': 16, 'ɕ': 16, 'ʑ': 16,
            'sh': 16, 'zh': 16, 'ch': 16, 'ts': 16, 'dz': 16,  # Add additional affricates
            'ʃː': 16, 'ʒː': 16, 'tʃː': 16, 'dʒː': 16,  # Add long variants
            
            # 17: Voiced dental fricative
            'ð': 17, 'dh': 17, 'ðː': 17,  # Add long variant
            
            # 18: Labiodental fricatives
            'f': 18, 'v': 18, 'ɱ': 18, 'fː': 18, 'vː': 18,  # Add long variants
            
            # 19: Alveolar stops, nasal + voiceless dental
            't': 19, 'd': 19, 'n': 19, 'θ': 19, 'th': 19, 'ɗ': 19, 'n̩': 19,
            'ɳ': 19, 'ṭ': 19, 'ḍ': 19, 
            'tʰ': 19, 'tː': 19, 'dː': 19, 'nː': 19, 'θː': 19,  # Add aspirated and long variants
            'ɾ̃': 19,  # Nasalized flap
            
            # 20: Velar consonants
            'k': 20, 'g': 20, 'ŋ': 20, 'ɲ': 20, 'x': 20, 'ɣ': 20, 'q': 20, 'ɢ': 20,
            'ɡ': 20, 'ng': 20, 'ŋ̍': 20,
            'kʰ': 20, 'kː': 20, 'gː': 20, 'ŋː': 20,  # Add aspirated and long variants
            
            # 21: Bilabial consonants
            'p': 21, 'b': 21, 'm': 21, 'ɓ': 21, 'm̩': 21,
            'pʰ': 21, 'pː': 21, 'bː': 21, 'mː': 21,  # Add aspirated and long variants
            'ʙ': 21,  # Bilabial trill
        }
        
        # Add viseme ID to name mapping for better readability
        self.viseme_id_to_name = {
            0: "SILENCE",
            1: "VOWEL_CENTRAL",
            2: "VOWEL_OPEN_BACK",
            3: "VOWEL_OPEN_MID_BACK",
            4: "VOWEL_MID",
            5: "VOWEL_RHOTIC",
            6: "VOWEL_CLOSE_FRONT",
            7: "VOWEL_CLOSE_BACK",
            8: "VOWEL_MID_BACK",
            9: "DIPHTHONG_AW",
            10: "DIPHTHONG_OY",
            11: "DIPHTHONG_AY",
            12: "CONSONANT_GLOTTAL",
            13: "CONSONANT_RHOTIC",
            14: "CONSONANT_LATERAL",
            15: "CONSONANT_ALVEOLAR_FRICATIVE",
            16: "CONSONANT_POSTALVEOLAR",
            17: "CONSONANT_DENTAL_VOICED",
            18: "CONSONANT_LABIODENTAL",
            19: "CONSONANT_ALVEOLAR_DENTAL",
            20: "CONSONANT_VELAR",
            21: "CONSONANT_BILABIAL"
        }

    def get_phoneme_features(self, phoneme):
        """
        Get the phonetic features for a given phoneme using panphon.
        
        Args:
            phoneme: The phoneme to get features for
            
        Returns:
            dict: A dictionary of phonetic features
        """
        # Handle special characters
        if phoneme in ['.', ' ', '-'] or not phoneme.strip():
            return {'is_silence': 1}
        
        # Get feature vector from panphon
        try:
            feature_vector = self.ft.word_to_vector_list(phoneme, numeric=True)
            
            # If panphon returned a valid feature vector
            if feature_vector and len(feature_vector) > 0:
                # Convert to dictionary with feature names
                feature_dict = {}
                for i, feature_name in enumerate(self.ft.names):
                    feature_dict[feature_name] = int(feature_vector[0][i])
                
                return feature_dict
            else:
                # No features returned by panphon
                return {'unknown': 1}
                
        except Exception as e:
            print(f"Warning: Error getting features for '{phoneme}': {e}")
            # Return minimal features on error
            return {'error': 1}

    def normalize_text(self, text):
        """
        Normalize text by removing non-alphanumeric characters and extra whitespace.
        Replaces apostrophes with blank spaces and preserves spaces for silences.
        
        Args:
            text: Text to normalize
            
        Returns:
            str: Normalized text with non-alphanumeric characters removed
        """
        if not text:
            return ""
        
        # Replace apostrophes with space first
        text = text.replace("'", " ")
        
        # Remove all non-alphanumeric characters except spaces
        normalized = self.alphanumeric_pattern.sub('', text)
        
        # Replace multiple spaces with a single space and strip
        normalized = ' '.join(normalized.split())
        
        return normalized

    def calculate_phoneme_distance(self, phoneme1, phoneme2, use_weights=None):
        """
        Calculate distance between two individual phonemes, with option to use weights.
        
        Args:
            phoneme1: First phoneme
            phoneme2: Second phoneme
            use_weights: Whether to use weighted features (defaults to self.use_weighted_distance)
            
        Returns:
            float: Distance value between 0.0 (identical) and 1.0 (maximally different)
            
        Raises:
            ValueError: If phoneme comparison fails due to missing or invalid data
        """
        # If use_weights is not specified, use the class setting
        if use_weights is None:
            use_weights = getattr(self, 'use_weighted_distance', False)
            
        # Special case: if either is empty or whitespace
        if not phoneme1.strip() or not phoneme2.strip():
            # Empty compared to anything else is maximally different
            return 1.0
            
        # If phonemes are identical, return 0 distance
        if phoneme1 == phoneme2:
            return 0.0
        
        try:
            # Use weighted or standard calculation based on parameter
            if use_weights and hasattr(self, 'feature_weights') and self.feature_weights:
                # Get feature vectors for the phonemes
                # Get raw feature vectors from panphon directly
                fv1 = self.ft.word_to_vector_list(phoneme1, numeric=True)
                fv2 = self.ft.word_to_vector_list(phoneme2, numeric=True)
                
                if not fv1 or not fv2:
                    print(f"Warning: Failed to compute features for phonemes '{phoneme1}' and/or '{phoneme2}'")
                    raise ValueError(f"Failed to compute features for phonemes '{phoneme1}' and/or '{phoneme2}'")
                
                # Extract vectors
                vec1 = fv1[0]
                vec2 = fv2[0]
                
                total_diff = 0.0
                total_weight = 0.0
                
                # Compare each feature with its weight
                for i, feature_name in enumerate(self.ft.names):
                    if feature_name in self.feature_weights:
                        diff = abs(vec1[i] - vec2[i])
                        weight = self.feature_weights[feature_name]
                        
                        total_diff += diff * weight
                        total_weight += weight
                
                # Calculate weighted average
                if total_weight > 0:
                    weighted_distance = total_diff / total_weight
                    return min(1.0, weighted_distance)  # Ensure it's within range
                else:
                    print(f"Warning: No common features found for phonemes '{phoneme1}' and '{phoneme2}'")
                    raise ValueError(f"No common features found for phonemes '{phoneme1}' and '{phoneme2}'")
            else:
                # Use standard panphon distance
                distance = self.dst.feature_edit_distance(phoneme1, phoneme2)
                
                # Calculate theoretical maximum distance based on feature count
                # Each panphon feature can contribute a maximum of 1.0 to the distance
                # There are 24 features in the standard panphon feature set
                max_theoretical_distance = len(self.ft.names)
                
                # Normalize to 0-1 range using theoretical maximum
                return min(1.0, distance / max_theoretical_distance)
                
        except Exception as e:
            print(f"Error calculating distance between '{phoneme1}' and '{phoneme2}': {e}")
            raise ValueError(f"Failed to calculate phoneme distance between '{phoneme1}' and '{phoneme2}': {e}")
    
    def calculate_sequence_distance(self, seq1, seq2, use_weights=None):
        """
        Calculate edit distance between two sequences (phonemes or visemes).
        
        Args:
            seq1: First sequence
            seq2: Second sequence
            use_weights: Whether to use weighted features (defaults to self.use_weighted_distance)
            
        Returns:
            tuple: (alignment, edit_distance)
        """
        # If use_weights is not specified, use the class setting
        if use_weights is None:
            use_weights = getattr(self, 'use_weighted_distance', False)
            
        # Create cost function with specified weighting
        def distance_cost(item1, item2):
            try:
                return self.calculate_phoneme_distance(item1, item2, use_weights=use_weights)
            except ValueError as e:
                # In sequence comparison context, fallback to maximum distance
                # This allows the overall alignment to succeed even if some pairs fail
                print(f"Using fallback distance (1.0) for '{item1}' and '{item2}': {e}")
                return 1.0
            
        # Use the unified alignment function
        return self.calculate_sequence_alignment(seq1, seq2, cost_function=distance_cost)
        
    # Legacy method names for backward compatibility
    def calculate_phonetic_distance(self, phoneme1, phoneme2):
        """Legacy method that calls calculate_phoneme_distance with current weighting"""
        return self.calculate_phoneme_distance(phoneme1, phoneme2)
        
    def calculate_weighted_phonetic_edit_distance(self, phoneme_seq1, phoneme_seq2):
        """Legacy method that calls calculate_sequence_distance with weighted=True"""
        _, edit_distance = self.calculate_sequence_distance(phoneme_seq1, phoneme_seq2, use_weights=True)
        return edit_distance
        
    def calculate_phonetic_alignment(self, seq1, seq2):
        """Legacy method that calls calculate_sequence_distance with weighted=False"""
        return self.calculate_sequence_distance(seq1, seq2, use_weights=False)
        
    def standard_phonetic_distance(self, phoneme1, phoneme2):
        """
        Calculate phonetic distance using the standard (non-weighted) approach.
        This ensures we have a clear distinction between standard and weighted distances.
        
        Args:
            phoneme1: First phoneme
            phoneme2: Second phoneme
            
        Returns:
            float: Distance value between 0.0 (identical) and 1.0 (maximally different)
            
        Raises:
            ValueError: If phoneme comparison fails due to missing or invalid data
        """
        # Special case: if either is empty or whitespace
        if not phoneme1.strip() or not phoneme2.strip():
            # Empty compared to anything else is maximally different
            return 1.0
        
        # Use standard panphon distance
        try:
            # Use panphon's feature edit distance
            distance = self.dst.feature_edit_distance(phoneme1, phoneme2)
            max_theoretical_distance = len(self.ft.names)
            
            # Normalize to 0-1 range (panphon distances can be larger)
            return min(1.0, distance / max_theoretical_distance)
        except Exception as e:
            print(f"Error calculating distance between '{phoneme1}' and '{phoneme2}': {e}")
            raise ValueError(f"Failed to calculate standard phonetic distance between '{phoneme1}' and '{phoneme2}': {e}")

    def calculate_sequence_alignment(self, seq1, seq2, cost_function=None, allow_weighted=False):
        """
        Unified sequence alignment function for both phonemes and visemes with support for weighted costs.
        
        Parameters:
        - seq1: First sequence (reference)
        - seq2: Second sequence (hypothesis)
        - cost_function: Optional function to calculate substitution cost (default: binary cost)
        - allow_weighted: Whether to allow weighted costs based on similarity matrix (for visemes)
        
        Returns:
        - alignment: List of (operation, seq1_item, seq2_item) tuples
        - edit_distance: Edit distance score
        """
        # Initialize matrix
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill first row and column
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill the matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # Calculate substitution cost
                if seq1[i-1] == seq2[j-1]:
                    # Exact match
                    cost = 0
                elif cost_function is not None:
                    # Use provided cost function
                    cost = cost_function(seq1[i-1], seq2[j-1])
                elif hasattr(self, 'use_weighted_distance') and self.use_weighted_distance and allow_weighted and hasattr(self, 'viseme_similarity_matrix'):
                    # Use weighted cost from similarity matrix for visemes
                    similarity = self.viseme_similarity_matrix.get((seq1[i-1], seq2[j-1]), 0)
                    cost = 1.0 - similarity
                else:
                    # Standard binary cost
                    cost = 1
                
                dp[i][j] = min(
                    dp[i-1][j] + 1,        # Deletion
                    dp[i][j-1] + 1,        # Insertion
                    dp[i-1][j-1] + cost    # Substitution
                )
        
        # Reconstruct alignment
        alignment = []
        i, j = m, n
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                # Calculate the cost that was used
                if seq1[i-1] == seq2[j-1]:
                    match_cost = 0
                elif cost_function is not None:
                    match_cost = cost_function(seq1[i-1], seq2[j-1])
                elif hasattr(self, 'use_weighted_distance') and self.use_weighted_distance and allow_weighted and hasattr(self, 'viseme_similarity_matrix'):
                    similarity = self.viseme_similarity_matrix.get((seq1[i-1], seq2[j-1]), 0)
                    match_cost = 1.0 - similarity
                else:
                    match_cost = 1
                
                if dp[i][j] == dp[i-1][j-1] + match_cost:
                    # Substitution or match
                    if seq1[i-1] == seq2[j-1]:
                        alignment.append(('match', seq1[i-1], seq2[j-1]))
                    else:
                        alignment.append(('substitute', seq1[i-1], seq2[j-1]))
                    i -= 1
                    j -= 1
                    continue
            
            if i > 0 and dp[i][j] == dp[i-1][j] + 1:
                # Deletion
                alignment.append(('delete', seq1[i-1], None))
                i -= 1
            else:
                # Insertion
                alignment.append(('insert', None, seq2[j-1]))
                j -= 1
        
        # Reverse alignment to get correct order
        alignment.reverse()
        
        return alignment, dp[m][n]

    def calculate_viseme_alignment(self, seq1, seq2):
        """
        Calculate alignment between two viseme sequences using dynamic programming
        
        Returns:
        - alignment: List of (operation, seq1_item, seq2_item) tuples
        - edit_distance: Edit distance score
        """
        # Use the unified alignment function with weighted option for visemes
        return self.calculate_sequence_alignment(seq1, seq2, allow_weighted=True)
    
    def text_to_phonemes(self, text):
        """Convert English text to phoneme sequence"""
        # Normalize text first to remove punctuation
        normalized_text = self.normalize_text(text)
        
        # If empty after normalization, return empty list
        if not normalized_text:
            return []
            
        processed_phonemes = []
        
        # Get raw IPA string and split by words
        raw_phonemes = self.phonemizer_backend.phonemize(
            [normalized_text], 
            separator=self.separator
        )[0]
        raw_words = raw_phonemes.split()
        
        # Process word by word
        for word_idx, word in enumerate(raw_words):
            # Add space between words (except before the first word)
            if word_idx > 0:
                processed_phonemes.append(' ')
                
            # Skip empty words
            if not word.strip():
                continue
            
            # Segment the IPA string into individual phonemes
            i = 0
            while i < len(word):
                # Try to match phonemes in mapping (longer sequences first)
                matched = False
                for phoneme_len in [3, 2, 1]:
                    if i + phoneme_len <= len(word):
                        candidate = word[i:i+phoneme_len]
                        if candidate in self.phoneme_to_viseme:
                            processed_phonemes.append(candidate)
                            i += phoneme_len
                            matched = True
                            break
                
                if not matched:
                    # If no match found, use individual character
                    processed_phonemes.append(word[i])
                    i += 1
        
        return processed_phonemes
    
    def evaluate_pair(self, reference, hypothesis):
        """
        Evaluate a single reference-hypothesis pair for phonetic and viseme similarity
        
        Parameters:
        - reference: Reference text (ground truth)
        - hypothesis: Hypothesis text (predicted)
        
        Returns:
        - Dictionary with evaluation results
        """
        # Normalize texts first
        normalized_reference = self.normalize_text(reference)
        normalized_hypothesis = self.normalize_text(hypothesis)
        
        # Convert texts to phoneme sequences
        try:
            ref_phonemes = self.text_to_phonemes(normalized_reference)
            hyp_phonemes = self.text_to_phonemes(normalized_hypothesis)        
            
            # Calculate phonetic alignment and edit distance
            alignment, edit_distance = self.calculate_phonetic_alignment(ref_phonemes, hyp_phonemes)
            
            # Convert phonemes to visemes using most appropriate mapping
            ref_visemes = [self.map_phoneme_to_viseme(p) for p in ref_phonemes]
            hyp_visemes = [self.map_phoneme_to_viseme(p) for p in hyp_phonemes]
            
            # Calculate viseme-level alignment and score
            viseme_alignment, viseme_edit_distance = self.calculate_viseme_alignment(ref_visemes, hyp_visemes)
            
            # Normalize viseme score (lower is better, convert to higher is better)
            max_len = max(len(ref_visemes), len(hyp_visemes))
            if max_len > 0:
                viseme_score = 1.0 - (viseme_edit_distance / max_len)
            else:
                viseme_score = 1.0
            
            # Store original texts and results
            results = {
                'reference': reference,
                'hypothesis': hypothesis,
                'ref_phonemes': ref_phonemes,
                'hyp_phonemes': hyp_phonemes,
                'phonetic_alignment': alignment,
                'phonetic_edit_distance': edit_distance,
                'ref_visemes': ref_visemes,
                'hyp_visemes': hyp_visemes,
                'viseme_alignment': viseme_alignment,
                'viseme_edit_distance': viseme_edit_distance,
                'viseme_score': viseme_score,
            }
            
            return results
            
        except Exception as e:
            print(f"  ERROR in evaluate_pair: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return a minimal result to avoid crashing
            return {
                'reference': reference,
                'hypothesis': hypothesis,
                'error': str(e),
                'viseme_score': 0.0,
                'phonetic_edit_distance': float('inf'),
            }

    def map_phoneme_to_viseme(self, phoneme, default_value=-1):
        """
        Map a phoneme to its corresponding viseme using direct mapping.
        
        Args:
            phoneme: The phoneme to map
            default_value: Value to return if phoneme isn't found (default: -1)
            
        Returns:
            int: The viseme ID (0-21) that corresponds to this phoneme,
                 or default_value if not found
        """
        try:
            # Try direct lookup first
            if phoneme in self.phoneme_to_viseme:
                return self.phoneme_to_viseme[phoneme]
            
            # Empty string or space is treated as silence
            if not phoneme.strip():
                return 0  # silence
            
            # If we still don't have a match, return the default value
            return default_value
        except Exception:
            # Return default value on any error
            return default_value

# Enhanced lip reading evaluator with weighted measurements
class WeightedLipReadingEvaluator(LipReadingEvaluator):
    """
    Enhanced lip reading evaluator that uses phonetic feature weighting
    for phoneme and viseme similarity calculations
    """
    
    def __init__(self, use_weighted_distance=True, weight_method="both"):
        """
        Initialize with option to use weighted distances
        
        Parameters:
        - use_weighted_distance: Whether to use phonetically weighted distances
        - weight_method: Method to calculate weights ("both", "entropy", or "distinctiveness")
        """
        # Initialize parent class
        super().__init__()
        
        # Set weighted distance flag
        self.use_weighted_distance = use_weighted_distance
        
        # Set weight calculation method
        self.weight_method = weight_method
        if weight_method not in ["both", "entropy", "distinctiveness"]:
            print(f"Warning: Invalid weight method '{weight_method}'. Using 'both' as default.")
            self.weight_method = "both"
        
        # Initialize viseme similarity matrix
        self.viseme_similarity_matrix = {}
        
        # Initialize feature weights
        self.feature_weights = {}
        
        # Calculate weights if enabled
        if use_weighted_distance:
            # Debug: Before weight calculation
            print("[Debug] Before weight calculation")
            
            # Calculate weights from scratch
            calculated_weights = self.calculate_phonetic_feature_weights()
            
            # Debug: After weight calculation
            print(f"[Debug] After weight calculation. Got {len(calculated_weights) if calculated_weights else 0} weights")
            
            # Store the weights
            self.feature_weights = calculated_weights
            
            # Debug: After assignment
            print(f"[Debug] After assignment. feature_weights has {len(self.feature_weights)} entries")
            
            # Calculate viseme similarity matrix
            self.viseme_similarity_matrix = self.calculate_viseme_similarity_matrix()
    
    def save_weights_to_file(self, file_path):
        """
        Save computed weights and similarity matrix to file for later use
        
        Parameters:
        - file_path: Path to save JSON file with weights
        """
        try:
            # Convert tuple keys to strings for JSON serialization
            serializable_matrix = {
                f"{k[0]},{k[1]}": v 
                for k, v in self.viseme_similarity_matrix.items()
            }
            
            data = {
                'feature_weights': self.feature_weights,
                'viseme_similarity_matrix': serializable_matrix
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            print(f"Saved weights and similarity matrix to {file_path}")
        except Exception as e:
            print(f"Error saving weights to file: {e}")

    def calculate_phonetic_feature_weights(self):
        """
        Calculate feature weights based on phonetic importance
        using entropy and visual distinctiveness.
        
        Returns:
            dict: Mapping of feature names to weight values (0-1)
        """
        print(f"Calculating phonetic feature weights using method: {self.weight_method}")
        weights = {}
        
        # Generate phoneme inventory directly from the complete mapping
        phoneme_inventory = list(self.phoneme_to_viseme.keys())
        
        # Remove silence markers and empty strings
        phoneme_inventory = [p for p in phoneme_inventory if p not in ['.', ' ', '-', '']]
        
        print(f"Using {len(phoneme_inventory)} phonemes from IPA inventory for weight calculation")
        
        # Get panphon feature names
        all_features = self.ft.names
        print(f"Analyzing {len(all_features)} panphon phonetic features")
        
        # Calculate entropy-based weights for each feature
        feature_entropies = {}
        for feature_idx, feature in enumerate(all_features):
            # Get all values this feature takes across the phoneme inventory
            feature_values = []
            for phoneme in phoneme_inventory:
                if phoneme in ['.', ' ', '-']:
                    continue
                    
                try:
                    # Get feature value from panphon directly
                    fv = self.ft.word_to_vector_list(phoneme, numeric=True)
                        
                    if fv and len(fv) > 0:
                        feature_values.append(fv[0][feature_idx])
                except Exception as e:
                    # Skip on error
                    continue
            
            if not feature_values:
                continue  # Skip if no values available
                
            # Count feature values (should be -1, 0, or 1 for most panphon features)
            value_counts = Counter(feature_values)
            
            # Calculate entropy
            total = len(feature_values)
            probabilities = [count/total for count in value_counts.values()]
            
            # Using numpy for stability in log calculations
            entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
            
            # Normalize by maximum possible entropy
            max_entropy = np.log2(len(value_counts)) if len(value_counts) > 1 else 1
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            feature_entropies[feature] = normalized_entropy
        
        # Calculate visual distinctiveness for each feature
        visual_distinctiveness = {}
        for feature_idx, feature in enumerate(all_features):
            # Group phonemes by feature value and viseme class
            feature_value_to_visemes = {}
            
            for phoneme in phoneme_inventory:
                if phoneme in ['.', ' ', '-']:
                    continue
                    
                try:
                    # Get feature value from panphon directly
                    fv = self.ft.word_to_vector_list(phoneme, numeric=True)
                        
                    if not fv or len(fv) == 0:
                        continue
                        
                    feature_value = fv[0][feature_idx]
                    viseme = self.map_phoneme_to_viseme(phoneme, -1)  # Use -1 for unknown
                    
                    if feature_value not in feature_value_to_visemes:
                        feature_value_to_visemes[feature_value] = set()
                    
                    feature_value_to_visemes[feature_value].add(viseme)
                except Exception:
                    # Skip on error
                    continue
            
            # If a feature value consistently maps to few visemes, it's visually distinctive
            total_distinctions = 0
            for viseme_set in feature_value_to_visemes.values():
                total_distinctions += len(viseme_set)
            
            total_values = len(feature_value_to_visemes)
            if total_values == 0:
                average_visemes_per_value = 0
            else:
                average_visemes_per_value = total_distinctions / total_values
            
            # Calculate visual distinctiveness
            total_viseme_classes = 22  # We use 22 viseme classes (0-21)
            
            # More distinctive = less visemes per feature value
            distinctiveness = 1.0 - (average_visemes_per_value / total_viseme_classes)
            distinctiveness = max(0.1, distinctiveness)  # Ensure minimum value
            
            visual_distinctiveness[feature] = distinctiveness
        
        # Combine entropy and visual distinctiveness based on selected method
        for feature in all_features:
            if feature in feature_entropies and feature in visual_distinctiveness:
                if self.weight_method == "both":
                    # Weight = entropy (information content) × visual distinctiveness
                    weights[feature] = feature_entropies[feature] * visual_distinctiveness[feature]
                elif self.weight_method == "entropy":
                    # Weight based only on entropy
                    weights[feature] = feature_entropies[feature]
                elif self.weight_method == "distinctiveness":
                    # Weight based only on visual distinctiveness
                    weights[feature] = visual_distinctiveness[feature]
                
                # Ensure minimum weight
                weights[feature] = max(0.1, weights[feature])
        
        # Print calculated weights for debugging
        print("\nPhonetic feature weights:")
        print("Feature | Entropy | Visual Distinctiveness | Final Weight")
        print("--------|---------|------------------------|-------------")
        for feature in sorted(weights.keys(), key=lambda f: weights[f], reverse=True):
            entropy = feature_entropies.get(feature, 0)
            distinctiveness = visual_distinctiveness.get(feature, 0)
            weight = weights[feature]
            print(f"{feature:8} | {entropy:.3f}  | {distinctiveness:.3f}                | {weight:.3f}")
        
        return weights
    
    def calculate_viseme_similarity_matrix(self):
        """
        Pre-calculate similarity between all viseme pairs based on
        phonetic features and phonetic feature weights.
        
        Returns:
            dict: Mapping of (viseme1, viseme2) tuples to similarity values
        """
        print("Calculating viseme similarity matrix...")
        similarity_matrix = {}
        
        # Get all viseme classes
        viseme_classes = sorted(set(self.map_phoneme_to_viseme(p) for p in self.phoneme_to_viseme.keys() if p not in ['.', ' ', '-', '']))
        print(f"Processing {len(viseme_classes)} viseme classes")
        
        # Get all phonemes for each viseme class - using all available phonemes without limiting
        viseme_to_phonemes = {}
        for viseme in viseme_classes:
            phonemes = [p for p in self.phoneme_to_viseme.keys() 
                       if self.map_phoneme_to_viseme(p) == viseme and p not in ['.', ' ', '-']]
            viseme_to_phonemes[viseme] = phonemes
            print(f"  Viseme {viseme} ({self.viseme_id_to_name.get(viseme, 'Unknown')}): {len(phonemes)} phonemes")
        
        # Calculate similarity between each pair of viseme classes
        for i, viseme1 in enumerate(viseme_classes):
            for viseme2 in viseme_classes:
                # Skip if already calculated (symmetrical)
                if (viseme1, viseme2) in similarity_matrix or (viseme2, viseme1) in similarity_matrix:
                    continue
                
                # Get representative phonemes
                phonemes1 = viseme_to_phonemes.get(viseme1, [])
                phonemes2 = viseme_to_phonemes.get(viseme2, [])
                
                if not phonemes1 or not phonemes2:
                    # If no phonemes for either viseme, set maximum distance
                    similarity = 0.0
                else:
                    # Calculate pairwise distances between all phonemes
                    distances = []
                    for p1 in phonemes1:
                        for p2 in phonemes2:
                            try:
                                # Use calculated distance with panphon features
                                distance = self.calculate_phoneme_distance(p1, p2, use_weights=True)
                                distances.append(distance)
                            except Exception as e:
                                print(f"Error comparing '{p1}' and '{p2}': {e}")
                    
                    # Calculate average distance
                    if distances:
                        avg_distance = sum(distances) / len(distances)
                        # Apply a non-linear transformation to better separate similar and dissimilar pairs
                        # (1 - x^2) gives more separation for small distances
                        similarity = 1.0 - (avg_distance ** 2)
                    else:
                        similarity = 0.0  # Default to no similarity if calculation failed
                
                # Store in matrix (symmetrical)
                similarity_matrix[(viseme1, viseme2)] = similarity
                similarity_matrix[(viseme2, viseme1)] = similarity
                
                # Same viseme should have perfect similarity
                if viseme1 == viseme2:
                    similarity_matrix[(viseme1, viseme2)] = 1.0
        
        # Print some examples
        print("\nViseme Similarity Examples:")
        print("Viseme Pair | Similarity")
        print("-----------|----------")
        
        # Print top 5 most similar different viseme pairs
        similar_pairs = [(v1, v2) for (v1, v2), sim in similarity_matrix.items() 
                        if v1 != v2 and sim > 0]
        for (v1, v2) in sorted(similar_pairs, key=lambda p: similarity_matrix[p], reverse=True)[:5]:
            sim = similarity_matrix[(v1, v2)]
            print(f"{v1} vs {v2} | {sim:.3f}")
        
        # Print 5 least similar pairs
        print("\nLeast Similar Viseme Pairs:")
        print("Viseme Pair | Similarity")
        print("-----------|----------")
        for (v1, v2) in sorted(similar_pairs, key=lambda p: similarity_matrix[p])[:5]:
            sim = similarity_matrix[(v1, v2)]
            print(f"{v1} vs {v2} | {sim:.3f}")
        
        return similarity_matrix
    
    def calculate_phonetic_distance(self, phoneme1, phoneme2):
        """
        Override parent method to calculate weighted phonetic distance between two phonemes
        using feature weights.
        
        Args:
            phoneme1: First phoneme
            phoneme2: Second phoneme
            
        Returns:
            float: Weighted distance value between 0.0 (identical) and 1.0 (maximally different)
        """
        # Call the new calculate_phoneme_distance method with current weighting
        return self.calculate_phoneme_distance(phoneme1, phoneme2)
    
    def calculate_weighted_phonetic_edit_distance(self, phoneme_seq1, phoneme_seq2):
        """
        Calculate phonetic edit distance between two phoneme sequences using weighted distance.
        
        Args:
            phoneme_seq1: First phoneme sequence
            phoneme_seq2: Second phoneme sequence
            
        Returns:
            float: Weighted edit distance value
        """
        # Define cost function that uses our class's weighted distance calculation
        def phonetic_weighted_cost(p1, p2):
            return self.calculate_phonetic_distance(p1, p2)
        
        # Use the unified alignment function with our weighted phonetic distance
        _, edit_distance = self.calculate_sequence_alignment(
            phoneme_seq1, phoneme_seq2, 
            cost_function=phonetic_weighted_cost
        )
        
        return edit_distance
        
    def evaluate_pair(self, reference, hypothesis):
        """
        Override parent method to evaluate a reference-hypothesis pair using weighted distance
        
        Args:
            reference: Reference text (ground truth)
            hypothesis: Hypothesis text (predicted)
            
        Returns:
            dict: Evaluation results
        """
        # Normalize texts first
        normalized_reference = self.normalize_text(reference)
        normalized_hypothesis = self.normalize_text(hypothesis)
        
        # Convert text to phonemes using normalized text
        ref_phonemes = self.text_to_phonemes(normalized_reference)
        hyp_phonemes = self.text_to_phonemes(normalized_hypothesis)
        
        # Convert phonemes to visemes
        ref_visemes = [self.map_phoneme_to_viseme(p) for p in ref_phonemes]
        hyp_visemes = [self.map_phoneme_to_viseme(p) for p in hyp_phonemes]
        
        # Calculate phonetic distances using appropriate method based on weighting setting
        if self.use_weighted_distance:
            # For weighted approach, use weighted sequence distance
            _, phonetic_distance = self.calculate_sequence_distance(ref_phonemes, hyp_phonemes, use_weights=True)
            phonetic_alignment = None  # We don't need the alignment for the weighted version
        else:
            # For standard approach, use standard sequence distance
            phonetic_alignment, phonetic_distance = self.calculate_sequence_distance(ref_phonemes, hyp_phonemes, use_weights=False)
        
        # Calculate viseme alignment
        alignment, edit_distance = self.calculate_viseme_alignment(ref_visemes, hyp_visemes)
        
        # Calculate normalized scores (0-1, higher is better)
        max_viseme_len = max(len(ref_visemes), len(hyp_visemes))
        max_phoneme_len = max(len(ref_phonemes), len(hyp_phonemes))
        
        if self.use_weighted_distance:
            # For weighted approach, use phonetically_weighted_viseme_score
            phonetically_weighted_viseme_score = 1.0 - (edit_distance / max_viseme_len if max_viseme_len > 0 else 0.0)
            score_key = 'phonetically_weighted_viseme_score'
            score_value = phonetically_weighted_viseme_score
        else:
            # For standard approach, use viseme_score
            viseme_score = 1.0 - (edit_distance / max_viseme_len if max_viseme_len > 0 else 0.0)
            score_key = 'viseme_score'
            score_value = viseme_score
        
        phonetic_alignment_score = 1.0 - (phonetic_distance / max_phoneme_len if max_phoneme_len > 0 else 0.0)
        
        # Return results
        results = {
            'ref_phonemes': ref_phonemes,
            'hyp_phonemes': hyp_phonemes,
            'ref_visemes': ref_visemes,
            'hyp_visemes': hyp_visemes,
            'phonetic_edit_distance': phonetic_distance,
            'phonetic_alignment_score': phonetic_alignment_score,
            'viseme_edit_distance': edit_distance,
            'viseme_alignment': alignment,
            score_key: score_value
        }
        
        return results
    
    def compare_standard_and_weighted(self, reference, hypothesis):
        """
        Evaluate a single reference-hypothesis pair with detailed metrics using both
        standard and weighted approaches for comparison.
        
        Parameters:
        - reference: Reference text (ground truth)
        - hypothesis: Hypothesis text (predicted)
        
        Returns:
        - Dictionary with evaluation results for both approaches
        """
        # Store original weighted distance setting (but preserve IPA setting)
        original_setting = self.use_weighted_distance
        
        # Normalize texts for consistency
        normalized_reference = self.normalize_text(reference)
        normalized_hypothesis = self.normalize_text(hypothesis)
        
        try:
            # Evaluate with standard distance (no weights)
            self.use_weighted_distance = False
            standard_results = self.evaluate_pair(normalized_reference, normalized_hypothesis)
            
            # Evaluate with weighted distance
            self.use_weighted_distance = True
            weighted_results = self.evaluate_pair(normalized_reference, normalized_hypothesis)
            
            # Restore original setting
            self.use_weighted_distance = original_setting
            
            # Combine results
            combined_results = {
                'reference': reference,
                'hypothesis': hypothesis,
                'ref_phonemes': standard_results.get('ref_phonemes', []),
                'hyp_phonemes': standard_results.get('hyp_phonemes', []),
                'standard': {
                    'phonetic_edit_distance': standard_results.get('phonetic_edit_distance', float('inf')),
                    'phonetic_alignment_score': standard_results.get('phonetic_alignment_score', 0.0),
                    'viseme_edit_distance': standard_results.get('viseme_edit_distance', float('inf')),
                    'viseme_score': standard_results.get('viseme_score', 0.0),
                },
                'weighted': {
                    'phonetic_edit_distance': weighted_results.get('phonetic_edit_distance', float('inf')),
                    'phonetic_alignment_score': weighted_results.get('phonetic_alignment_score', 0.0),
                    'viseme_edit_distance': weighted_results.get('viseme_edit_distance', float('inf')),
                    'phonetically_weighted_viseme_score': weighted_results.get('phonetically_weighted_viseme_score', 0.0),
                }
            }
            
            return combined_results
            
        except Exception as e:
            print(f"ERROR in compare_standard_and_weighted: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Restore original setting
            self.use_weighted_distance = original_setting
            
            # Return a minimal result
            return {
                'reference': reference,
                'hypothesis': hypothesis,
                'error': str(e),
                'standard': {'viseme_score': 0.0},
                'weighted': {'phonetically_weighted_viseme_score': 0.0}
            }
    
    def calculate_additional_metrics(self, example_pairs):
        """
        Calculate additional metrics for a list of reference-hypothesis pairs
        
        Parameters:
        - example_pairs: List of (reference, hypothesis) tuples
        
        Returns:
        - Dictionary with additional metrics (WER, CER, etc.)
        """
        import nltk
        try:
            from nltk.translate import meteor_score
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            # Ensure necessary NLTK data is available
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                print("Downloading NLTK punkt tokenizer...")
                nltk.download('punkt', quiet=True)
        except ImportError:
            print("Warning: NLTK is not fully available. Some metrics may not be calculated.")
        
        try:
            from difflib import SequenceMatcher
            from rouge_score import rouge_scorer
        except ImportError:
            print("Warning: rouge_score is not available. ROUGE metrics will not be calculated.")
        
        metrics = {}
        per_example_metrics = []
        
        # Extract references and hypotheses
        references = [ref for ref, _ in example_pairs]
        hypotheses = [hyp for _, hyp in example_pairs]
        
        # Character Error Rate (CER)
        try:
            cer_values = []
            for ref, hyp in example_pairs:
                if len(ref) > 0:
                    char_error = nltk.edit_distance(ref, hyp)
                    cer = char_error / len(ref)
                    cer_values.append(cer)
                else:
                    cer_values.append(1.0 if hyp else 0.0)
                
            total_chars = sum(len(ref) for ref in references)
            char_errors = sum(nltk.edit_distance(ref, hyp) for ref, hyp in example_pairs)
            cer = char_errors / total_chars if total_chars > 0 else 0
            metrics['character_error_rate'] = cer
        except Exception as e:
            print(f"Warning: Could not calculate CER: {e}")
            cer_values = [0.0] * len(example_pairs)
        
        # Word Error Rate (WER)
        try:
            wer_values = []
            for ref, hyp in example_pairs:
                ref_words = ref.split()
                hyp_words = hyp.split()
                if ref_words:
                    word_error = nltk.edit_distance(ref_words, hyp_words)
                    wer = word_error / len(ref_words)
                    wer_values.append(wer)
                else:
                    wer_values.append(1.0 if hyp_words else 0.0)
            
            ref_words = [ref.split() for ref in references]
            hyp_words = [hyp.split() for hyp in hypotheses]
            total_words = sum(len(words) for words in ref_words)
            word_errors = sum(nltk.edit_distance(ref, hyp) for ref, hyp in zip(ref_words, hyp_words))
            wer = word_errors / total_words if total_words > 0 else 0
            metrics['word_error_rate'] = wer
        except Exception as e:
            print(f"Warning: Could not calculate WER: {e}")
            wer_values = [0.0] * len(example_pairs)
        
        # Word similarity using SequenceMatcher
        try:
            word_similarities = [SequenceMatcher(None, ref, hyp).ratio() for ref, hyp in example_pairs]
            metrics['word_similarity'] = np.mean(word_similarities)
        except Exception as e:
            print(f"Warning: Could not calculate word similarity: {e}")
            word_similarities = [0.0] * len(example_pairs)
        
        # METEOR score
        try:
            meteor_scores = []
            for ref, hyp in example_pairs:
                ref_tokens = [ref.split()]  # METEOR expects a list of reference tokenized sentences
                hyp_tokens = hyp.split()
                score = meteor_score.meteor_score(ref_tokens, hyp_tokens)
                meteor_scores.append(score)
            metrics['meteor_score'] = np.mean(meteor_scores)
        except Exception as e:
            print(f"Warning: Could not calculate METEOR score: {e}")
            meteor_scores = [0.0] * len(example_pairs)
        
        # ROUGE scores
        try:
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            rouge1_scores = []
            rouge2_scores = []
            rougeL_scores = []
            
            for ref, hyp in example_pairs:
                scores = scorer.score(ref, hyp)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
            
            metrics['rouge1_score'] = np.mean(rouge1_scores)
            metrics['rouge2_score'] = np.mean(rouge2_scores)
            metrics['rougeL_score'] = np.mean(rougeL_scores)
        except Exception as e:
            print(f"Warning: Could not calculate ROUGE scores: {e}")
            rouge1_scores = [0.0] * len(example_pairs)
            rouge2_scores = [0.0] * len(example_pairs)
            rougeL_scores = [0.0] * len(example_pairs)
        
        # BLEU scores
        try:
            # Get smoothing function for BLEU
            smoothie = SmoothingFunction().method1
            
            # Sentence BLEU
            sentence_bleu_scores = []
            for ref, hyp in example_pairs:
                ref_tokens = [ref.split()]  # BLEU expects a list of reference tokenized sentences
                hyp_tokens = hyp.split()
                
                # Skip empty sequences
                if not hyp_tokens or not ref_tokens[0]:
                    sentence_bleu_scores.append(0.0)
                    continue
                
                # Use smoothing function to handle cases with no n-gram overlaps
                score = sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=smoothie)
                sentence_bleu_scores.append(score)
            
            metrics['sentence_bleu_score'] = np.mean(sentence_bleu_scores) if sentence_bleu_scores else 0
        except Exception as e:
            print(f"Warning: Could not calculate BLEU scores: {e}")
            sentence_bleu_scores = [0.0] * len(example_pairs)
        
        # Try to calculate BERTScore if available
        try:
            import torch
            import bert_score
            P, R, F1 = bert_score.score(hypotheses, references, lang="en", verbose=False)
            
            bertscore_p = [p.item() for p in P]
            bertscore_r = [r.item() for r in R]
            bertscore_f1 = [f1.item() for f1 in F1]
            
            metrics['bertscore_precision'] = float(torch.mean(P).item())
            metrics['bertscore_recall'] = float(torch.mean(R).item())
            metrics['bertscore_f1'] = float(torch.mean(F1).item())
        except Exception as e:
            print(f"Warning: Could not calculate BERTScore: {e}")
            bertscore_p = [0.0] * len(example_pairs)
            bertscore_r = [0.0] * len(example_pairs)
            bertscore_f1 = [0.0] * len(example_pairs)
        
        # Try to calculate semantic similarity if available
        try:
            from sentence_transformers import SentenceTransformer
            semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            ref_embeddings = semantic_model.encode(references)
            hyp_embeddings = semantic_model.encode(hypotheses)
            
            semantic_similarities = []
            for i in range(len(references)):
                ref_emb = ref_embeddings[i]
                hyp_emb = hyp_embeddings[i]
                similarity = np.dot(ref_emb, hyp_emb) / (np.linalg.norm(ref_emb) * np.linalg.norm(hyp_emb))
                semantic_similarities.append(float(similarity))
            
            metrics['semantic_similarity'] = float(np.mean(semantic_similarities))
            
            # Calculate semantic WER as 1 - semantic similarity (simplified)
            semantic_wer = 1.0 - metrics['semantic_similarity']
            metrics['semantic_wer'] = float(semantic_wer)
        except Exception as e:
            print(f"Warning: Could not calculate semantic metrics: {e}")
            semantic_similarities = [0.0] * len(example_pairs)
            
        # Store per-example metrics
        for i, (ref, hyp) in enumerate(example_pairs):
            example_metrics = {
                'reference': ref,
                'hypothesis': hyp,
                'cer': cer_values[i] if i < len(cer_values) else 0.0,
                'wer': wer_values[i] if i < len(wer_values) else 0.0,
                'word_similarity': word_similarities[i] if i < len(word_similarities) else 0.0,
                'meteor_score': meteor_scores[i] if i < len(meteor_scores) else 0.0,
                'rouge1_score': rouge1_scores[i] if i < len(rouge1_scores) else 0.0,
                'rouge2_score': rouge2_scores[i] if i < len(rouge2_scores) else 0.0,
                'rougeL_score': rougeL_scores[i] if i < len(rougeL_scores) else 0.0,
                'sentence_bleu_score': sentence_bleu_scores[i] if i < len(sentence_bleu_scores) else 0.0
            }
            
            # Add bertscore if available
            if i < len(bertscore_p):
                example_metrics['bertscore_precision'] = bertscore_p[i]
                example_metrics['bertscore_recall'] = bertscore_r[i]
                example_metrics['bertscore_f1'] = bertscore_f1[i]
            
            # Add semantic similarity if available
            if i < len(semantic_similarities):
                example_metrics['semantic_similarity'] = semantic_similarities[i]
                example_metrics['semantic_wer'] = 1.0 - semantic_similarities[i]
            
            per_example_metrics.append(example_metrics)
        
        # Save metrics to separate file
        self.save_additional_metrics_to_file(metrics)
        
        return metrics, per_example_metrics
    
    def save_additional_metrics_to_file(self, metrics, file_path=None):
        """
        Save additional metrics to a separate file
        
        Parameters:
        - metrics: Dictionary with metrics to save
        - file_path: Path to save file (default: metrics.txt in current directory)
        """
        if file_path is None:
            file_path = 'metrics.txt'
        
        try:
            with open(file_path, 'w') as f:
                f.write("=== ADDITIONAL METRICS ===\n\n")
                for metric, value in metrics.items():
                    f.write(f"{metric}: {value:.4f}\n")
            
            print(f"Saved additional metrics to {file_path}")
        except Exception as e:
            print(f"Error saving additional metrics: {e}")

    def analyze_json_dataset_with_comparisons(self, json_file, output_file=None, max_samples=None, include_all_metrics=False, export_csv=False):
        """
        Analyze a dataset with both standard and weighted approaches for comparison
        
        Parameters:
        - json_file: Path to JSON file with reference-hypothesis pairs
        - output_file: Path to save analysis results (optional)
        - max_samples: Maximum number of samples to process (default: all)
        - include_all_metrics: Whether to include additional metrics
        - export_csv: Whether to export results to CSV
        
        Returns:
        - Dictionary with analysis results
        """
        try:
            # Load JSON data
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            print(f"Loaded data from {json_file}")
            
            # Create pairs for analysis based on the JSON structure
            example_pairs = []
            
            # Special case: Dictionary with "ref"/"hypo" lists
            if isinstance(data, dict) and 'ref' in data and 'hypo' in data:
                if isinstance(data['ref'], list) and isinstance(data['hypo'], list):
                    # Pair up matching indices from ref and hypo lists
                    num_pairs = min(len(data['ref']), len(data['hypo']))
                    print(f"Found parallel ref/hypo lists with {num_pairs} pairs")
                    
                    for i in range(num_pairs):
                        example_pairs.append((data['ref'][i], data['hypo'][i]))
            
            # If we don't have pairs yet, try other formats
            if not example_pairs:
                # Try standard list of dicts format
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            if 'reference' in item and 'hypothesis' in item:
                                example_pairs.append((item['reference'], item['hypothesis']))
                            elif 'ref' in item and 'hyp' in item:
                                example_pairs.append((item['ref'], item['hyp']))
                            elif 'ground_truth' in item and 'prediction' in item:
                                example_pairs.append((item['ground_truth'], item['prediction']))
                        elif isinstance(item, list) and len(item) >= 2:
                            example_pairs.append((item[0], item[1]))
                
                # Try top-level dict format
                elif isinstance(data, dict):
                    if 'reference' in data and 'hypothesis' in data:
                        example_pairs.append((data['reference'], data['hypothesis']))
                    elif 'ref' in data and 'hyp' in data:
                        example_pairs.append((data['ref'], data['hyp']))
            
            if not example_pairs:
                print("Could not extract reference-hypothesis pairs from the JSON file.")
                return None
            
            # Limit number of samples if specified
            if max_samples is not None and len(example_pairs) > max_samples:
                print(f"Limiting analysis to {max_samples} samples (out of {len(example_pairs)} total)")
                example_pairs = example_pairs[:max_samples]
            
            print(f"Processing {len(example_pairs)} reference-hypothesis pairs")
            
            # Process all examples with both approaches
            all_results = []
            
            # Add progress reporting
            print("Starting evaluation with both standard and weighted approaches...")
            total_examples = len(example_pairs)
            
            for i, (ref, hyp) in enumerate(example_pairs):
                if i % 10 == 0 or i == total_examples - 1:
                    print(f"Evaluating example {i+1}/{total_examples} ({(i+1)/total_examples*100:.1f}%)")
                
                # Evaluate with both approaches
                results = self.compare_standard_and_weighted(ref, hyp)
                all_results.append(results)
            
            # Calculate summary statistics
            summary = {
                'num_examples': len(all_results),
                'standard': {
                    'avg_viseme_score': np.mean([r['standard']['viseme_score'] for r in all_results]),
                    'avg_phonetic_distance': np.mean([r['standard']['phonetic_edit_distance'] for r in all_results 
                                                    if r['standard']['phonetic_edit_distance'] != float('inf')]),
                    'avg_phonetic_score': np.mean([r['standard']['phonetic_alignment_score'] for r in all_results]),
                },
                'weighted': {
                    'avg_phonetically_weighted_viseme_score': np.mean([r['weighted']['phonetically_weighted_viseme_score'] for r in all_results]),
                    'avg_phonetic_distance': np.mean([r['weighted']['phonetic_edit_distance'] for r in all_results
                                                    if r['weighted']['phonetic_edit_distance'] != float('inf')]),
                    'avg_phonetic_score': np.mean([r['weighted']['phonetic_alignment_score'] for r in all_results]),
                }
            }
            
            # Calculate difference statistics
            viseme_score_diffs = [r['weighted']['phonetically_weighted_viseme_score'] - r['standard']['viseme_score'] 
                          for r in all_results]
            phonetic_score_diffs = [r['weighted']['phonetic_alignment_score'] - r['standard']['phonetic_alignment_score'] 
                          for r in all_results]
            
            summary['comparison'] = {
                'avg_score_difference': np.mean(viseme_score_diffs),
                'max_score_improvement': max(viseme_score_diffs),
                'avg_phonetic_score_difference': np.mean(phonetic_score_diffs),
                'max_phonetic_score_improvement': max(phonetic_score_diffs),
                'percent_viseme_improved': sum(1 for d in viseme_score_diffs if d > 0) / len(viseme_score_diffs) * 100,
                'percent_viseme_unchanged': sum(1 for d in viseme_score_diffs if d == 0) / len(viseme_score_diffs) * 100,
                'percent_viseme_worse': sum(1 for d in viseme_score_diffs if d < 0) / len(viseme_score_diffs) * 100,
                'percent_phonetic_improved': sum(1 for d in phonetic_score_diffs if d > 0) / len(phonetic_score_diffs) * 100,
                'percent_phonetic_unchanged': sum(1 for d in phonetic_score_diffs if d == 0) / len(phonetic_score_diffs) * 100,
                'percent_phonetic_worse': sum(1 for d in phonetic_score_diffs if d < 0) / len(phonetic_score_diffs) * 100,
            }
            
            # Add additional metrics if requested
            per_example_metrics = []
            if include_all_metrics:
                print("\nCalculating additional metrics...")
                additional_metrics, per_example_metrics = self.calculate_additional_metrics(example_pairs)
                summary['additional_metrics'] = additional_metrics
                
                # Copy the core metrics to summary level
                for key, value in additional_metrics.items():
                    summary[key] = value
            
            # Print summary
            print("\n=== ANALYSIS SUMMARY ===")
            print(f"Total examples: {summary['num_examples']}")
            print("\nStandard approach:")
            print(f"  Average viseme score: {summary['standard']['avg_viseme_score']:.3f}")
            print(f"  Average phonetic score: {summary['standard']['avg_phonetic_score']:.3f}")
            print(f"  Average phonetic distance: {summary['standard']['avg_phonetic_distance']:.3f}")
            print("\nWeighted approach:")
            print(f"  Average phonetically-weighted viseme score: {summary['weighted']['avg_phonetically_weighted_viseme_score']:.3f}")
            print(f"  Average phonetic score: {summary['weighted']['avg_phonetic_score']:.3f}")
            print(f"  Average phonetic distance: {summary['weighted']['avg_phonetic_distance']:.3f}")
            print("\nComparison:")
            print(f"  Average score difference: {summary['comparison']['avg_score_difference']:.3f}")
            print(f"  Maximum score improvement: {summary['comparison']['max_score_improvement']:.3f}")
            print(f"  Viseme examples improved: {summary['comparison']['percent_viseme_improved']:.1f}%")
            print(f"  Average phonetic score difference: {summary['comparison']['avg_phonetic_score_difference']:.3f}")
            print(f"  Maximum phonetic score improvement: {summary['comparison']['max_phonetic_score_improvement']:.3f}")
            print(f"  Phonetic examples improved: {summary['comparison']['percent_phonetic_improved']:.1f}%")
            
            # Print additional metrics if requested
            if include_all_metrics and 'additional_metrics' in summary:
                print("\nAdditional Metrics:")
                for metric, value in summary['additional_metrics'].items():
                    print(f"  {metric}: {value:.4f}")
            
            # Save results if output file specified
            if output_file:
                # Create directory if needed
                os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
                
                results_to_save = {
                    'summary': summary,
                    'examples': all_results
                }
                
                # Add per-example metrics if calculated
                if per_example_metrics:
                    # Merge the per-example metrics with the existing results
                    for i, (result, metrics) in enumerate(zip(all_results, per_example_metrics)):
                        if i < len(all_results):
                            # Add additional metrics to each example
                            result['metrics'] = metrics
                            
                            # Add backward compatibility mapping for 'phonemes'
                            if 'ref_phonemes' in result and 'phonemes' not in result:
                                result['phonemes'] = result['ref_phonemes']
                
                with open(output_file, 'w') as f:
                    json.dump(results_to_save, f, indent=2)
                
                print(f"\nSaved analysis results to {output_file}")
                
                # Save a separate summary-only file
                summary_file = os.path.join(os.path.dirname(os.path.abspath(output_file)), 'summary.json')
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2)
                print(f"Saved summary metrics to {summary_file}")
            
            # Export to CSV if requested
            if export_csv:
                csv_file = os.path.join(os.path.dirname(os.path.abspath(output_file if output_file else 'results.json')), 'results.csv')
                self.export_results_to_csv(all_results, per_example_metrics, csv_file)
                print(f"Saved detailed results to CSV: {csv_file}")
            
            return {
                'summary': summary,
                'results': all_results
            }
            
        except Exception as e:
            print(f"Error analyzing dataset: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def export_results_to_csv(self, results, additional_metrics=None, csv_file='results.csv'):
        """
        Export evaluation results to a CSV file with proper column separation
        
        Parameters:
        - results: List of evaluation result dictionaries
        - additional_metrics: List of additional metrics dictionaries (optional)
        - csv_file: Path to save the CSV file
        """
        import csv
        
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(os.path.abspath(csv_file)), exist_ok=True)
            
            # Define CSV columns - only include necessary columns
            columns = [
                'reference', 
                'hypothesis',
                'ref_phonemes',
                'hyp_phonemes',
                'std_viseme_score',
                'wgt_phonetically_weighted_viseme_score',
                'score_difference',
                'std_phonetic_distance',
                'wgt_phonetic_distance'
            ]
            
            # Add additional metric columns if available
            additional_metric_keys = []
            if additional_metrics and len(additional_metrics) > 0:
                for metrics in additional_metrics:
                    if metrics:  # Check if metrics exist for this sample
                        for key in metrics.keys():
                            # Only add metrics that aren't already included and aren't reference/hypothesis
                            if key not in ['reference', 'hypothesis'] and key not in columns and key not in additional_metric_keys:
                                additional_metric_keys.append(key)
                
                # Sort additional metrics alphabetically for consistency
                additional_metric_keys.sort()
                columns.extend(additional_metric_keys)
            
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                
                # Write each result row
                for i, result in enumerate(results):
                    # Make sure we have valid data for this result
                    if not isinstance(result, dict) or 'standard' not in result or 'weighted' not in result:
                        continue
                        
                    # Basic row information
                    row = {
                        'reference': result.get('reference', ''),
                        'hypothesis': result.get('hypothesis', ''),
                        'ref_phonemes': ' '.join(result.get('ref_phonemes', [])) if result.get('ref_phonemes') else '',
                        'hyp_phonemes': ' '.join(result.get('hyp_phonemes', [])) if result.get('hyp_phonemes') else ''
                    }
                    
                    # Standard metrics
                    std = result.get('standard', {})
                    row['std_viseme_score'] = round(std.get('viseme_score', 0), 4) if 'viseme_score' in std else ''
                    
                    # Handle special case for infinity
                    std_ped = std.get('phonetic_edit_distance', None)
                    if std_ped == float('inf'):
                        row['std_phonetic_distance'] = 'inf'
                    elif std_ped is not None:
                        row['std_phonetic_distance'] = round(std_ped, 4)
                    else:
                        row['std_phonetic_distance'] = ''
                    
                    # Weighted metrics
                    wgt = result.get('weighted', {})
                    row['wgt_phonetically_weighted_viseme_score'] = round(wgt.get('phonetically_weighted_viseme_score', 0), 4) if 'phonetically_weighted_viseme_score' in wgt else ''
                    
                    # Handle special case for infinity
                    wgt_ped = wgt.get('phonetic_edit_distance', None)
                    if wgt_ped == float('inf'):
                        row['wgt_phonetic_distance'] = 'inf'
                    elif wgt_ped is not None:
                        row['wgt_phonetic_distance'] = round(wgt_ped, 4)
                    else:
                        row['wgt_phonetic_distance'] = ''
                    
                    # Calculate score difference
                    if 'viseme_score' in std and 'phonetically_weighted_viseme_score' in wgt:
                        row['score_difference'] = round(wgt['phonetically_weighted_viseme_score'] - std['viseme_score'], 4)
                    else:
                        row['score_difference'] = ''
                    
                    # Add additional metrics if available
                    if additional_metrics and i < len(additional_metrics) and additional_metrics[i]:
                        metric_data = additional_metrics[i]
                        for key in additional_metric_keys:
                            value = metric_data.get(key, '')
                            # Format numeric values
                            if isinstance(value, (int, float)):
                                row[key] = round(value, 4)
                            else:
                                row[key] = value
                    
                    writer.writerow(row)
                    
            print(f"Successfully exported {len(results)} results to {csv_file}")
            
        except Exception as e:
            print(f"Error exporting results to CSV: {e}")
            import traceback
            traceback.print_exc()

    def visualize_results(self, analysis_results, output_dir):
        """Create visualizations for the analysis results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all results from the analysis
        all_results = analysis_results.get('results', [])
        summary = analysis_results.get('summary', {})
        
        if not all_results:
            print("No results to visualize")
            return
        
        print(f"Creating visualizations in {output_dir}...")
        
        # Create confusion matrices
        self._plot_confusion_matrices(all_results, output_dir)
        
        print("Visualizations complete")
    
    def _plot_confusion_matrices(self, all_results, output_dir):
        """Plot viseme confusion matrices for standard and weighted approaches"""
        # Collect viseme substitutions from alignments
        std_substitutions = []
        wgt_substitutions = []
        
        for result in all_results:
            ref_phonemes = result.get('ref_phonemes', [])
            hyp_phonemes = result.get('hyp_phonemes', [])
            
            # Ensure we have phonemes for processing
            if not ref_phonemes:
                ref_phonemes = self.text_to_phonemes(result['reference'])
            if not hyp_phonemes:
                hyp_phonemes = self.text_to_phonemes(result['hypothesis'])
            
            # Convert phonemes to visemes - using consistent phoneme-to-viseme mapping
            ref_visemes = [self.map_phoneme_to_viseme(p) for p in ref_phonemes]
            hyp_visemes = [self.map_phoneme_to_viseme(p) for p in hyp_phonemes]
            
            # Get alignment and find substitutions
            # First for standard approach
            self.use_weighted_distance = False
            alignment, _ = self.calculate_viseme_alignment(ref_visemes, hyp_visemes)
            for op, ref_v, hyp_v in alignment:
                if op == 'substitute':
                    std_substitutions.append((ref_v, hyp_v))
            
            # Then for weighted approach
            self.use_weighted_distance = True
            alignment, _ = self.calculate_viseme_alignment(ref_visemes, hyp_visemes)
            for op, ref_v, hyp_v in alignment:
                if op == 'substitute':
                    wgt_substitutions.append((ref_v, hyp_v))
        
        # Restore original setting
        self.use_weighted_distance = getattr(self, 'use_weighted_distance', True)
        
        # Get unique viseme classes from the data
        all_visemes = list(range(22))  # We use 22 viseme classes (0-21)
        
        # Use the viseme_id_to_name from the parent class for better readability
        if hasattr(self, 'viseme_id_to_name'):
            viseme_names = self.viseme_id_to_name
        else:
            # Fallback to basic naming
            viseme_names = {v: f"Viseme {v}" for v in all_visemes}
        
        # Create shortened labels for the axis to avoid overlap
        viseme_labels = [f"{v}" for v in all_visemes]
        
        # Create confusion matrices
        if std_substitutions and all_visemes:
            # Standard approach matrix   
            std_true, std_pred = zip(*std_substitutions)
            std_cm = confusion_matrix(std_true, std_pred, labels=all_visemes)
            
            # Normalize safely - avoid division by zero
            row_sums = std_cm.sum(axis=1)
            std_cm_norm = np.zeros_like(std_cm, dtype=float)
            for i in range(len(row_sums)):
                if row_sums[i] > 0:  # Only normalize if row sum is positive
                    std_cm_norm[i] = std_cm[i] / row_sums[i]
            
            # Weighted approach matrix
            wgt_true, wgt_pred = zip(*wgt_substitutions)
            wgt_cm = confusion_matrix(wgt_true, wgt_pred, labels=all_visemes)
            
            # Normalize safely - avoid division by zero
            row_sums = wgt_cm.sum(axis=1)
            wgt_cm_norm = np.zeros_like(wgt_cm, dtype=float)
            for i in range(len(row_sums)):
                if row_sums[i] > 0:  # Only normalize if row sum is positive
                    wgt_cm_norm[i] = wgt_cm[i] / row_sums[i]
            
            # Add diagnostic information about viseme class distribution
            missing_std_visemes = [v for i, v in enumerate(all_visemes) if row_sums[i] == 0]
            if missing_std_visemes:
                print(f"Note: Some viseme classes were not found in reference data: {missing_std_visemes}")
                print("This is normal if your dataset doesn't contain examples of these viseme classes.")
                missing_names = [self.viseme_id_to_name.get(v, f"Class {v}") for v in missing_std_visemes]
                print(f"Missing classes: {', '.join(missing_names)}")
            
            # Calculate difference matrix
            diff_cm = wgt_cm_norm - std_cm_norm
            
            # Define colormaps
            std_cmap = plt.cm.Blues
            wgt_cmap = plt.cm.Blues
            diff_cmap = sns.diverging_palette(240, 10, as_cmap=True)
            
            # Helper function to create and save a confusion matrix heatmap
            def create_heatmap(matrix, title, filename, cmap, vmin=0, vmax=1, center=None):
                plt.figure(figsize=(16, 12))
                
                # Split figure for main plot and legend
                gs = plt.GridSpec(1, 2, width_ratios=[3, 1])
                
                # Main heatmap axis
                ax_heatmap = plt.subplot(gs[0])
                ax_legend = plt.subplot(gs[1])
                
                # Create the heatmap
                sns.heatmap(matrix, annot=True, fmt='.2f', cmap=cmap, square=True, 
                           vmin=vmin, vmax=vmax, center=center,
                           annot_kws={"size": 8}, xticklabels=viseme_labels, 
                           yticklabels=viseme_labels, ax=ax_heatmap)
                
                # Set labels
                ax_heatmap.set_title(title)
                ax_heatmap.set_xlabel('Predicted Viseme')
                ax_heatmap.set_ylabel('True Viseme')
                
                # Add legend in the right panel
                ax_legend.axis('off')
                legend_text = "Viseme Reference:\n\n"
                for v in all_visemes:
                    legend_text += f"{v}: {viseme_names.get(v, 'Unknown')}\n"
                ax_legend.text(0, 0.5, legend_text, va='center', fontsize=10)
                
                # Save plot
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, filename), dpi=300)
                plt.close()
            
            # Create the three different confusion matrix visualizations
            create_heatmap(
                std_cm_norm, 
                'Standard Approach - Viseme Confusion Matrix',
                'standard_confusion_matrix.png',
                std_cmap
            )
            
            create_heatmap(
                wgt_cm_norm, 
                'Weighted Approach - Viseme Confusion Matrix',
                'weighted_confusion_matrix.png',
                wgt_cmap
            )
            
            create_heatmap(
                diff_cm, 
                'Difference (Weighted - Standard) Confusion Matrix',
                'difference_confusion_matrix.png',
                diff_cmap,
                vmin=-0.5,
                vmax=0.5,
                center=0
            )
    
    def standard_phonetic_distance(self, phoneme1, phoneme2):
        """
        Calculate phonetic distance using the standard (non-weighted) approach.
        This ensures we have a clear distinction between standard and weighted distances.
        
        Args:
            phoneme1: First phoneme
            phoneme2: Second phoneme
            
        Returns:
            float: Distance value between 0.0 (identical) and 1.0 (maximally different)
            
        Raises:
            ValueError: If phoneme comparison fails due to missing or invalid data
        """
        # Special case: if either is empty or whitespace
        if not phoneme1.strip() or not phoneme2.strip():
            # Empty compared to anything else is maximally different
            return 1.0
        
        # Use standard panphon distance
        try:
            # Use panphon's feature edit distance
            distance = self.dst.feature_edit_distance(phoneme1, phoneme2)
            max_theoretical_distance = len(self.ft.names)
            # Normalize to 0-1 range (panphon distances can be larger)
            return min(1.0, max_theoretical_distance / max_theoretical_distance)
        except Exception as e:
            print(f"Error calculating distance between '{phoneme1}' and '{phoneme2}': {e}")
            raise ValueError(f"Failed to calculate standard phonetic distance between '{phoneme1}' and '{phoneme2}': {e}")

def main():
    """Main function for demonstrating and comparing different alignment approaches"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Phonetically-Weighted Viseme Scoring')
    parser.add_argument('--json', type=str, help='JSON file with reference-hypothesis pairs')
    parser.add_argument('--weights', type=str, help='Path to save computed weights (will not load from this file)')
    parser.add_argument('--save_dir', type=str, default='viseme_output', help='Directory to save all outputs (results, plots, weights)')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to process')
    parser.add_argument('--compare', action='store_true', help='Run comparison examples')
    parser.add_argument('-all', action='store_true', help='Calculate all metrics including WER, CER, and others')
    parser.add_argument('--csv', action='store_true', help='Export results to CSV file')
    parser.add_argument('--weight-method', type=str, choices=['both', 'entropy', 'distinctiveness'], default='both',
                        help='Method to calculate feature weights: both (default), entropy-only, or distinctiveness-only')
    parser.add_argument('--compare-methods', action='store_true', help='Compare results using different weight methods')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Define output paths
    if args.weights:
        weights_path = args.weights
    else:
        method_suffix = f"_{args.weight_method}" if args.weight_method != "both" else ""
        weights_path = os.path.join(args.save_dir, f'viseme_weights{method_suffix}.json')
    
    results_path = os.path.join(args.save_dir, f'results{method_suffix}.json')
    metrics_path = os.path.join(args.save_dir, f'metrics{method_suffix}.txt')
    
    if args.compare_methods:
        # Run analysis with all three weight methods and compare
        print("\n=== COMPARING DIFFERENT WEIGHT METHODS ===")
        methods = ['both', 'entropy', 'distinctiveness']
        method_results = {}
        example_pairs = []
        
        # Load pairs from JSON if provided, or use sample examples
        if args.json:
            try:
                with open(args.json, 'r') as f:
                    data = json.load(f)
                
                # Extract pairs depending on the format
                if isinstance(data, dict) and 'ref' in data and 'hypo' in data:
                    pairs = list(zip(data['ref'], data['hypo']))
                elif isinstance(data, list):
                    pairs = [(item.get('reference', item.get('ref', '')), 
                             item.get('hypothesis', item.get('hyp', '')))
                             for item in data if isinstance(item, dict)]
                
                if pairs:
                    example_pairs = pairs[:20]  # Limit to 20 examples
            except Exception as e:
                print(f"Error loading JSON: {e}")
                example_pairs = []
        
        # If no pairs from JSON, use sample examples
        if not example_pairs:
            example_pairs = [
                ("Hello world", "Hello word"),
                ("Please pass the salt", "Please pass the fault"),
                ("Bring me a glass of water", "Bring me a glass of vodka"),
                ("Meet me at five", "Meet me at nine"),
                ("I enjoy boys playing outside", "I enjoy toys playing outside"),
            ]
        
        # Compare results for each method
        print(f"\nAnalyzing {len(example_pairs)} example pairs with three different weight methods")
        
        for method in methods:
            print(f"\n--- EVALUATING WITH {method.upper()} WEIGHTS ---")
            evaluator = WeightedLipReadingEvaluator(weight_method=method)
            
            # Save the weights
            method_suffix = f"_{method}" if method != "both" else ""
            method_weights_path = os.path.join(args.save_dir, f'viseme_weights{method_suffix}.json')
            evaluator.save_weights_to_file(method_weights_path)
            
            # Calculate scores for all examples
            scores = []
            for ref, hyp in example_pairs:
                result = evaluator.compare_standard_and_weighted(ref, hyp)
                scores.append({
                    'reference': ref,
                    'hypothesis': hyp,
                    'standard_score': result['standard']['viseme_score'],
                    'weighted_score': result['weighted']['phonetically_weighted_viseme_score'],
                    'improvement': result['weighted']['phonetically_weighted_viseme_score'] - result['standard']['viseme_score']
                })
            
            # Calculate average scores
            avg_standard = sum(s['standard_score'] for s in scores) / len(scores)
            avg_weighted = sum(s['weighted_score'] for s in scores) / len(scores)
            avg_improvement = sum(s['improvement'] for s in scores) / len(scores)
            
            method_results[method] = {
                'scores': scores,
                'avg_standard': avg_standard,
                'avg_weighted': avg_weighted,
                'avg_improvement': avg_improvement
            }
            
            print(f"Average standard score: {avg_standard:.3f}")
            print(f"Average weighted score: {avg_weighted:.3f}")
            print(f"Average improvement: {avg_improvement:.3f}")
            
            # Save detailed results to JSON
            details_path = os.path.join(args.save_dir, f'method_comparison_{method}.json')
            with open(details_path, 'w') as f:
                json.dump({'scores': scores, 'summary': {
                    'avg_standard': avg_standard,
                    'avg_weighted': avg_weighted,
                    'avg_improvement': avg_improvement
                }}, f, indent=2)
        
        # Summarize and compare
        print("\n=== WEIGHT METHOD COMPARISON SUMMARY ===")
        print("Method       | Std Score | Weighted Score | Improvement")
        print("-------------|-----------|---------------|------------")
        for method in methods:
            res = method_results[method]
            print(f"{method:12} | {res['avg_standard']:.3f}    | {res['avg_weighted']:.3f}       | {res['avg_improvement']:.3f}")
        
        # Save overall comparison to JSON
        comparison_path = os.path.join(args.save_dir, 'weight_method_comparison.json')
        with open(comparison_path, 'w') as f:
            json.dump({method: {
                'avg_standard': res['avg_standard'],
                'avg_weighted': res['avg_weighted'],
                'avg_improvement': res['avg_improvement']
            } for method, res in method_results.items()}, f, indent=2)
        
        print(f"\nSaved comparison results to {comparison_path}")
        print("Individual method results saved to method_comparison_*.json files")
        
        return
    
    # Create weighted evaluator with the specified weight method
    evaluator = WeightedLipReadingEvaluator(weight_method=args.weight_method)
    
    # Debug: Check feature weights
    print(f"Initial feature weights: {len(evaluator.feature_weights) if hasattr(evaluator, 'feature_weights') and evaluator.feature_weights else 'Not set'}")
    
    # Print phoneme conversion approach
    print("\n=== USING IPA CONVERSION ===")
    print(f"Using weight method: {args.weight_method}")
    
    print("Note: Weights are now always calculated from scratch")
    
    # Quick conversion test with a sentence containing 'oy' diphthong
    test_sentence = "I enjoy toys and boys playing outside."
    phonemes = evaluator.text_to_phonemes(test_sentence)
    print(f"Test conversion: '{test_sentence}'")
    print(f"Phonemes: {' '.join(phonemes)}")

    # Add detailed debugging of each phoneme and its viseme mapping
    print("\nDetailed phoneme-to-viseme mapping:")
    for p in phonemes:
        viseme_id = evaluator.map_phoneme_to_viseme(p)
        if viseme_id == 0:
            print(f"  '{p}' -> {viseme_id} (SILENCE)")
        else:
            print(f"  '{p}' -> {viseme_id} ({evaluator.viseme_id_to_name.get(viseme_id, 'Unknown')})")

    # Show all visemes
    visemes = [evaluator.map_phoneme_to_viseme(p) for p in phonemes]
    print(f"Visemes: {' '.join(str(v) for v in visemes)}")

    # Count silence visemes
    silence_count = sum(1 for v in visemes if v == 0)
    print(f"Silence visemes: {silence_count} out of {len(visemes)} ({silence_count/len(visemes)*100:.1f}%)")
    print()
    
    # Save weights
    if hasattr(evaluator, 'feature_weights') and evaluator.feature_weights:
        print(f"Feature weights before saving: {len(evaluator.feature_weights)} entries")
        evaluator.save_weights_to_file(weights_path)
        print(f"Saved computed weights to {weights_path}")
    else:
        print(f"No feature weights available to save. Type: {type(evaluator.feature_weights)}")
        print(f"Feature weights content: {evaluator.feature_weights}")
    
    # Run comparison examples if requested
    if args.compare:
        example_pairs = [
            ("Hello world", "Hello word"),   # Small difference
            ("Please pass the salt", "Please pass the fault"),   # Visually similar sounds
            ("Bring me a glass of water", "Bring me a glass of vodka"),  # Visually similar 'w' vs 'v'
            ("Meet me at five", "Meet me at nine"),  # 'f' vs 'n' (different visemes)
            ("I enjoy boys playing outside", "I enjoy toys playing outside"),  # 'b' vs 't' with 'oy' diphthong
        ]
        
        print(f"\n=== Comparing Standard vs. Phonetically-Weighted Viseme Scores (Method: {args.weight_method}) ===\n")
        
        for ref, hyp in example_pairs:
            print(f"\nReference: '{ref}'")
            print(f"Hypothesis: '{hyp}'")
            
            # Evaluate with both approaches
            results = evaluator.compare_standard_and_weighted(ref, hyp)
            
            # Print results
            print(f"Standard viseme score: {results['standard']['viseme_score']:.3f}")
            print(f"Phonetically-weighted viseme score: {results['weighted']['phonetically_weighted_viseme_score']:.3f}")
            viseme_diff = results['weighted']['phonetically_weighted_viseme_score'] - results['standard']['viseme_score']
            print(f"Difference: {viseme_diff:.3f} ({'better' if viseme_diff > 0 else 'worse' if viseme_diff < 0 else 'same'})")
    
    # Process JSON file if provided
    if args.json:
        results = evaluator.analyze_json_dataset_with_comparisons(
            args.json, 
            output_file=results_path,
            max_samples=args.max_samples,
            include_all_metrics=args.all,
            export_csv=args.csv
        )
        
        if results:
            # If all metrics were requested, save them
            if args.all and 'summary' in results and 'additional_metrics' in results['summary']:
                evaluator.save_additional_metrics_to_file(
                    results['summary']['additional_metrics'], 
                    file_path=metrics_path
                )
            
            print("\nGenerating visualizations...")
            evaluator.visualize_results(results, output_dir=args.save_dir)
            print(f"\nAnalysis complete! Using IPA conversion with {args.weight_method} weighting.")
            print(f"All outputs saved to: {args.save_dir}")


if __name__ == "__main__":
    main() 