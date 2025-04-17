import numpy as np
import math
from collections import Counter
from g2p_en import G2p  # Grapheme-to-phoneme converter
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch  # Add this import for Patch
from sklearn.manifold import TSNE
import networkx as nx
import os
import time
import json
from datetime import datetime
import panphon.distance  # Library for phonological features
import panphon.featuretable
from difflib import SequenceMatcher


class LipReadingEvaluator:
    """Comprehensive lip reading evaluation system with information-theoretic weights"""
    
    def __init__(self, load_from_file=None):
        """Initialize the evaluator with phoneme mappings and converters"""
        # Initialize G2P converter
        self.g2p = G2p()
        
        # Define comprehensive phoneme-to-viseme mapping
        self.phoneme_to_viseme = {
            # Bilabial consonants
            'p': 'bilabial', 'b': 'bilabial', 'm': 'bilabial',
            # Labiodental consonants
            'f': 'labiodental', 'v': 'labiodental',
            # Dental consonants
            'θ': 'dental', 'ð': 'dental',
            # Alveolar consonants
            't': 'alveolar', 'd': 'alveolar', 's': 'alveolar', 
            'z': 'alveolar', 'n': 'alveolar', 'l': 'alveolar',
            # Post-alveolar consonants
            'ʃ': 'post-alveolar', 'ʒ': 'post-alveolar', 
            'tʃ': 'post-alveolar', 'dʒ': 'post-alveolar',
            # Palatal
            'j': 'palatal', 'ɲ': 'palatal',
            # Velar consonants
            'k': 'velar', 'g': 'velar', 'ŋ': 'velar', 'w': 'velar',
            # Glottal consonants
            'h': 'glottal', 'ʔ': 'glottal',
            # Retroflex
            'r': 'retroflex', 'ɹ': 'retroflex', 'ɾ': 'retroflex',
            # Vowels by position
            # Front vowels
            'i': 'front_vowel', 'ɪ': 'front_vowel', 'e': 'front_vowel', 
            'ɛ': 'front_vowel', 'æ': 'front_vowel',
            # Central vowels
            'ə': 'central_vowel', 'ʌ': 'central_vowel', 'ɜ': 'central_vowel',
            'ɝ': 'central_vowel', 'ɚ': 'central_vowel',
            # Back vowels
            'u': 'back_vowel', 'ʊ': 'back_vowel', 'o': 'back_vowel', 
            'ɔ': 'back_vowel', 'ɑ': 'back_vowel', 'ɒ': 'back_vowel',
            # Diphthongs
            'eɪ': 'diphthong', 'aɪ': 'diphthong', 'ɔɪ': 'diphthong', 
            'aʊ': 'diphthong', 'oʊ': 'diphthong', 'ɪə': 'diphthong',
            'ʊə': 'diphthong', 'eə': 'diphthong',
            # Special characters
            '.': 'silence', ' ': 'silence', '-': 'silence'
        }
        
        if load_from_file:
            # Load phoneme features from file if provided
            self.load_phoneme_features(load_from_file)
        else:
            # Define default phoneme features (used as fallback)
            self.phoneme_features = self.generate_default_phoneme_features()
            
            # Try to generate phoneme features using panphon (preferred method)
            try:
                self.generate_phoneme_features_from_panphon()
            except ImportError:
                print("Note: panphon not available for automatic phoneme feature generation")
                print("Using predefined feature set instead")
        
        # Calculate feature weights based on information theory
        self.feature_weights = self.calculate_information_theoretic_weights()
        
        # Pre-calculate phoneme similarity matrix based on information-theoretic approach
        self.similarity_cache = {}

    def generate_default_phoneme_features(self):
        """Generate default phoneme features as fallback"""
        return {
            # Bilabial consonants
            'p': {'place': 1, 'manner': 'stop', 'voiced': 0},
            'b': {'place': 1, 'manner': 'stop', 'voiced': 1},
            'm': {'place': 1, 'manner': 'nasal', 'voiced': 1},
            # Labiodental consonants
            'f': {'place': 2, 'manner': 'fricative', 'voiced': 0},
            'v': {'place': 2, 'manner': 'fricative', 'voiced': 1},
            # Dental consonants
            'θ': {'place': 3, 'manner': 'fricative', 'voiced': 0},
            'ð': {'place': 3, 'manner': 'fricative', 'voiced': 1},
            # Alveolar consonants
            't': {'place': 4, 'manner': 'stop', 'voiced': 0},
            'd': {'place': 4, 'manner': 'stop', 'voiced': 1},
            's': {'place': 4, 'manner': 'fricative', 'voiced': 0},
            'z': {'place': 4, 'manner': 'fricative', 'voiced': 1},
            'n': {'place': 4, 'manner': 'nasal', 'voiced': 1},
            'l': {'place': 4, 'manner': 'liquid', 'voiced': 1},
            # Post-alveolar consonants
            'ʃ': {'place': 5, 'manner': 'fricative', 'voiced': 0},
            'ʒ': {'place': 5, 'manner': 'fricative', 'voiced': 1},
            'tʃ': {'place': 5, 'manner': 'affricate', 'voiced': 0},
            'dʒ': {'place': 5, 'manner': 'affricate', 'voiced': 1},
            # Palatal consonants
            'j': {'place': 5.5, 'manner': 'approximant', 'voiced': 1},
            'ɲ': {'place': 5.5, 'manner': 'nasal', 'voiced': 1},
            # Velar consonants
            'k': {'place': 6, 'manner': 'stop', 'voiced': 0},
            'g': {'place': 6, 'manner': 'stop', 'voiced': 1},
            'ŋ': {'place': 6, 'manner': 'nasal', 'voiced': 1},
            'w': {'place': 6, 'manner': 'approximant', 'voiced': 1},
            # Glottal consonants
            'h': {'place': 7, 'manner': 'fricative', 'voiced': 0},
            'ʔ': {'place': 7, 'manner': 'stop', 'voiced': 0},
            # Retroflex
            'r': {'place': 4.5, 'manner': 'liquid', 'voiced': 1, 'rhotic': 1},
            'ɹ': {'place': 4.5, 'manner': 'approximant', 'voiced': 1, 'rhotic': 1},
            'ɾ': {'place': 4.5, 'manner': 'flap', 'voiced': 1, 'rhotic': 1},
            
            # Vowels
            # Front vowels
            'i': {'height': 1, 'backness': 1, 'rounded': 0, 'tense': 1},
            'ɪ': {'height': 1.5, 'backness': 1, 'rounded': 0, 'tense': 0},
            'e': {'height': 2, 'backness': 1, 'rounded': 0, 'tense': 1},
            'ɛ': {'height': 3, 'backness': 1, 'rounded': 0, 'tense': 0},
            'æ': {'height': 3.5, 'backness': 1, 'rounded': 0, 'tense': 0},
            # Central vowels
            'ə': {'height': 2.5, 'backness': 2, 'rounded': 0, 'tense': 0},
            'ʌ': {'height': 3, 'backness': 2, 'rounded': 0, 'tense': 0},
            'ɜ': {'height': 3, 'backness': 2, 'rounded': 0, 'tense': 0},
            'ɝ': {'height': 3, 'backness': 2, 'rounded': 0, 'tense': 0, 'rhotic': 1},
            'ɚ': {'height': 2.5, 'backness': 2, 'rounded': 0, 'tense': 0, 'rhotic': 1},
            # Back vowels
            'u': {'height': 1, 'backness': 3, 'rounded': 1, 'tense': 1},
            'ʊ': {'height': 1.5, 'backness': 3, 'rounded': 1, 'tense': 0},
            'o': {'height': 2, 'backness': 3, 'rounded': 1, 'tense': 1},
            'ɔ': {'height': 3, 'backness': 3, 'rounded': 1, 'tense': 0},
            'ɑ': {'height': 4, 'backness': 3, 'rounded': 0, 'tense': 1},
            'ɒ': {'height': 4, 'backness': 3, 'rounded': 1, 'tense': 0},
            
            # Diphthongs
            'eɪ': {'start_height': 2, 'start_backness': 1, 'end_height': 1.5, 'end_backness': 1},
            'aɪ': {'start_height': 4, 'start_backness': 2, 'end_height': 1.5, 'end_backness': 1},
            'ɔɪ': {'start_height': 3, 'start_backness': 3, 'end_height': 1.5, 'end_backness': 1},
            'aʊ': {'start_height': 4, 'start_backness': 2, 'end_height': 1.5, 'end_backness': 3},
            'oʊ': {'start_height': 2, 'start_backness': 3, 'end_height': 1.5, 'end_backness': 3},
            'ɪə': {'start_height': 1.5, 'start_backness': 1, 'end_height': 2.5, 'end_backness': 2},
            'ʊə': {'start_height': 1.5, 'start_backness': 3, 'end_height': 2.5, 'end_backness': 2},
            'eə': {'start_height': 2, 'start_backness': 1, 'end_height': 2.5, 'end_backness': 2}
        }
    
    def generate_phoneme_features_from_panphon(self):
        """Generate phoneme features using panphon library as primary source"""
        try:
            # Initialize panphon feature table and distance calculator
            ft = panphon.featuretable.FeatureTable()
            dst = panphon.distance.Distance()
            
            # Create new feature dictionary based on panphon
            new_features = {}
            
            # Get all phonemes from our viseme mapping
            phonemes = list(self.phoneme_to_viseme.keys())
            
            # Process each phoneme
            for phoneme in phonemes:
                # Skip special characters and complex multi-character phonemes
                if phoneme in ['.', ' ', '-']:
                    continue
                    
                # Skip complex phonemes except common ones we know panphon can handle
                if len(phoneme) > 1 and not (phoneme in ['tʃ', 'dʒ', 'θ', 'ð', 'ʃ', 'ʒ', 'ŋ']):
                    # For diphthongs and other complex phonemes, keep default features
                    if phoneme in self.phoneme_features:
                        new_features[phoneme] = self.phoneme_features[phoneme]
                    continue
                
                try:
                    # Get binary feature vector from panphon
                    feature_vector = ft.word_to_vector_list(phoneme, numeric=True)
                    
                    if not feature_vector:
                        continue
                        
                    feature_vector = feature_vector[0]  # Take first phoneme's features
                    
                    # Convert to dictionary with feature names
                    feature_dict = {}
                    for i, feature_name in enumerate(ft.names):
                        feature_dict[feature_name] = int(feature_vector[i])
                    
                    # Add high-level classifications based on panphon features
                    
                    # Determine if vowel
                    is_vowel = feature_dict.get('syl', 0) == 1
                    
                    if is_vowel:
                        # Vowel features
                        feature_dict['is_vowel'] = 1
                        
                        # Height (high, mid, low)
                        if feature_dict.get('hi', 0) == 1:
                            feature_dict['height'] = 1  # High
                        elif feature_dict.get('lo', 0) == 1:
                            feature_dict['height'] = 4  # Low
                        else:
                            feature_dict['height'] = 2.5  # Mid
                            
                        # Backness (front, central, back)
                        if feature_dict.get('bck', 0) == 1:
                            feature_dict['backness'] = 3  # Back
                        elif feature_dict.get('fr', 0) == 1:
                            feature_dict['backness'] = 1  # Front
                        else:
                            feature_dict['backness'] = 2  # Central
                            
                        # Roundedness
                        feature_dict['rounded'] = feature_dict.get('rnd', 0)
                    else:
                        # Consonant features
                        feature_dict['is_consonant'] = 1
                        
                        # Place of articulation
                        if feature_dict.get('lab', 0) == 1:
                            if feature_dict.get('dnt', 0) == 1:
                                feature_dict['place'] = 2  # Labiodental
                            else:
                                feature_dict['place'] = 1  # Bilabial
                        elif feature_dict.get('dnt', 0) == 1:
                            feature_dict['place'] = 3  # Dental
                        elif feature_dict.get('alv', 0) == 1:
                            feature_dict['place'] = 4  # Alveolar
                        elif feature_dict.get('pla', 0) == 1:
                            feature_dict['place'] = 5  # Post-alveolar/Palatal
                        elif feature_dict.get('vel', 0) == 1:
                            feature_dict['place'] = 6  # Velar
                        elif feature_dict.get('glt', 0) == 1:
                            feature_dict['place'] = 7  # Glottal
                        
                        # Manner of articulation
                        if feature_dict.get('nas', 0) == 1:
                            feature_dict['manner'] = 'nasal'
                        elif feature_dict.get('stp', 0) == 1:
                            feature_dict['manner'] = 'stop'
                        elif feature_dict.get('frc', 0) == 1:
                            feature_dict['manner'] = 'fricative'
                        elif feature_dict.get('lat', 0) == 1:
                            feature_dict['manner'] = 'liquid'
                        elif feature_dict.get('flp', 0) == 1:
                            feature_dict['manner'] = 'flap'
                        else:
                            feature_dict['manner'] = 'approximant'
                        
                        # Voicing
                        feature_dict['voiced'] = feature_dict.get('voi', 0)
                        
                    # Preserve any existing features for this phoneme
                    if phoneme in self.phoneme_features:
                        # Merge with existing features, preferring panphon values
                        merged_dict = self.phoneme_features[phoneme].copy()
                        merged_dict.update(feature_dict)
                        feature_dict = merged_dict
                    
                    # Store the new feature dictionary
                    new_features[phoneme] = feature_dict
                    
                except Exception as e:
                    # Fallback to default features if available
                    if phoneme in self.phoneme_features:
                        new_features[phoneme] = self.phoneme_features[phoneme]
                    print(f"Warning: Could not process phoneme '{phoneme}' with panphon: {e}")
            
            # Update features with panphon-derived ones, keeping defaults for any missing phonemes
            for phoneme in self.phoneme_features:
                if phoneme not in new_features:
                    new_features[phoneme] = self.phoneme_features[phoneme]
            
            self.phoneme_features = new_features
            print(f"Successfully generated phoneme features for {len(new_features)} phonemes using panphon")
            return True
            
        except Exception as e:
            print(f"Error generating phoneme features with panphon: {e}")
            return False
    
    def enrich_features_with_panphon(self):
        """Augment existing phoneme features using panphon library (legacy method)"""
        try:
            # Initialize panphon feature table
            ft = panphon.featuretable.FeatureTable()
            
            # Create a distance metric
            dst = panphon.distance.Distance()
            
            # Process each phoneme in our inventory
            for phoneme in list(self.phoneme_features.keys()):
                # Skip complex phonemes like diphthongs
                if len(phoneme) > 1 and not (phoneme in ['tʃ', 'dʒ', 'θ', 'ð', 'ʃ', 'ʒ', 'ŋ']):
                    continue
                
                try:
                    # Get binary feature vector
                    features = ft.word_to_vector_list(phoneme, numeric=True)
                    
                    if features:
                        features = features[0]  # Take first phoneme's features
                        
                        # Add these features to our existing feature set
                        feature_dict = {}
                        for i, feature_name in enumerate(ft.names):
                            # Convert to binary or ternary values
                            feature_dict[feature_name] = features[i]
                        
                        # Merge with existing features
                        self.phoneme_features[phoneme].update(feature_dict)
                except:
                    # Skip if phoneme not recognized by panphon
                    pass
                    
            print(f"Enriched phoneme features with panphon data for {len(self.phoneme_features)} phonemes")
        except Exception as e:
            print(f"Could not use panphon for feature enrichment: {e}")
    
    def load_phoneme_features(self, filename):
        """Load phoneme features from a JSON file"""
        try:
            with open(filename, 'r') as f:
                self.phoneme_features = json.load(f)
            print(f"Loaded phoneme features from {filename}")
        except Exception as e:
            print(f"Error loading phoneme features: {e}")
            # Fall back to default features definition
            self.__init__()
    
    def save_phoneme_features(self, filename):
        """Save phoneme features to a JSON file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.phoneme_features, f, indent=2)
            print(f"Saved phoneme features to {filename}")
        except Exception as e:
            print(f"Error saving phoneme features: {e}")
    
    def calculate_information_theoretic_weights(self):
        """
        Calculate feature weights based on information theory principles
        using entropy and visual distinctiveness.
        """
        weights = {}
        phoneme_inventory = list(self.phoneme_features.keys())
        
        # For each feature type (place, manner, voiced, etc.)
        all_features = set()
        for features in self.phoneme_features.values():
            all_features.update(features.keys())
        
        # Calculate entropy-based weights for each feature
        feature_entropies = {}
        for feature in all_features:
            # Get all values this feature takes across the phoneme inventory
            feature_values = []
            for phoneme in phoneme_inventory:
                if feature in self.phoneme_features[phoneme]:
                    feature_values.append(self.phoneme_features[phoneme][feature])
            
            if not feature_values:
                continue  # Skip if no values available
                
            # Handle numeric vs categorical features
            if all(isinstance(val, (int, float)) for val in feature_values):
                # For numeric features, discretize into bins
                min_val = min(feature_values)
                max_val = max(feature_values)
                if max_val > min_val:
                    bins = 5
                    bin_size = (max_val - min_val) / bins
                    binned_values = [int((val - min_val) / bin_size) for val in feature_values]
                    value_counts = Counter(binned_values)
                else:
                    value_counts = Counter(feature_values)
            else:
                # For categorical features, count occurrences directly
                value_counts = Counter(feature_values)
            
            # Calculate entropy
            total = len(feature_values)
            probabilities = [count/total for count in value_counts.values()]
            entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
            
            # Normalize by maximum possible entropy
            max_entropy = math.log2(len(value_counts)) if len(value_counts) > 1 else 1
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            feature_entropies[feature] = normalized_entropy
        
        # Calculate visual distinctiveness for each feature
        visual_distinctiveness = {}
        for feature in all_features:
            # Group phonemes by feature value and viseme class
            feature_value_to_visemes = {}
            
            for phoneme in phoneme_inventory:
                if feature not in self.phoneme_features[phoneme]:
                    continue
                    
                feature_value = self.phoneme_features[phoneme][feature]
                viseme = self.phoneme_to_viseme.get(phoneme, 'other')
                
                if feature_value not in feature_value_to_visemes:
                    feature_value_to_visemes[feature_value] = set()
                
                feature_value_to_visemes[feature_value].add(viseme)
            
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
            total_viseme_classes = len(set(self.phoneme_to_viseme.values()))
            
            # More distinctive = less visemes per feature value
            distinctiveness = 1.0 - (average_visemes_per_value / total_viseme_classes)
            distinctiveness = max(0.1, distinctiveness)  # Ensure minimum value
            
            visual_distinctiveness[feature] = distinctiveness
        
        # Combine entropy and visual distinctiveness
        for feature in all_features:
            if feature in feature_entropies and feature in visual_distinctiveness:
                # Weight = entropy (information content) × visual distinctiveness
                weights[feature] = feature_entropies[feature] * visual_distinctiveness[feature]
                # Ensure minimum weight
                weights[feature] = max(0.1, weights[feature])
        
        # Print calculated weights for debugging
        print("Information-theoretic feature weights:")
        print("Feature | Entropy | Visual Distinctiveness | Final Weight")
        print("--------|---------|------------------------|-------------")
        for feature in sorted(weights.keys()):
            entropy = feature_entropies.get(feature, 0)
            distinctiveness = visual_distinctiveness.get(feature, 0)
            weight = weights[feature]
            print(f"{feature:8} | {entropy:.3f}  | {distinctiveness:.3f}                | {weight:.3f}")
        
        return weights
    
    def analyze_json_dataset(self, json_file, output_dir=None, max_samples=None):
        """
        Analyze a dataset of reference-hypothesis pairs from a JSON file
        
        Parameters:
        - json_file: Path to JSON file with format 
                     [{"reference": "text1", "hypothesis": "text2"}, ...]
                     or {"ref": [ref1, ref2, ...], "hyp": [hyp1, hyp2, ...]}
        - output_dir: Directory to save output visualizations (default: create timestamped dir)
        - max_samples: Maximum number of samples to process (default: all)
        
        Returns:
        - Dictionary with analysis results and visualization paths
        """
        try:
            # Load JSON data
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            print(f"Loaded data from {json_file}")
            
            # Print diagnostic information about the JSON structure
            print(f"\n=== JSON STRUCTURE DIAGNOSIS ===")
            print(f"Data type: {type(data)}")
            
            if isinstance(data, list):
                print(f"List with {len(data)} items")
                print(f"First 3 items (of {len(data)}):")
                for i in range(min(3, len(data))):
                    print(f"  [{i}] Type: {type(data[i])}")
                    print(f"      Value: {data[i]}")
            elif isinstance(data, dict):
                print(f"Keys in dictionary: {list(data.keys())}")
                for key in list(data.keys())[:3]:  # Show first 3 keys
                    print(f"  ['{key}'] Type: {type(data[key])}")
                    value_str = str(data[key])
                    if len(value_str) > 100:
                        value_str = value_str[:100] + "..."
                    print(f"      Value: {value_str}")
            else:
                print(f"Data: {data}")
            
            print(f"=== END OF DIAGNOSIS ===\n")
            
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
                print("For your specific format, please try again with the temp file created or")
                print("convert your data to the format: [{\"reference\": \"text1\", \"hypothesis\": \"text1\"}, ...]")
                return None
            
            # Limit number of samples if specified
            if max_samples is not None and len(example_pairs) > max_samples:
                print(f"Limiting analysis to {max_samples} samples (out of {len(example_pairs)} total)")
                example_pairs = example_pairs[:max_samples]
            
            print(f"Processing {len(example_pairs)} reference-hypothesis pairs")
            
            # Create output directory
            if output_dir is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_name = os.path.splitext(os.path.basename(json_file))[0]
                output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                         "plots", f"{base_name}_analysis_{timestamp}")
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Initialize phoneme cache
            if not hasattr(self, 'phoneme_cache'):
                self.phoneme_cache = {}
            
            # Process all examples
            all_results = []
            
            # Add progress reporting
            print("Starting evaluation...")
            total_examples = len(example_pairs)
            progress_interval = max(1, total_examples // 10)  # Report progress every 10%
            
            for i, (ref, hyp) in enumerate(example_pairs):
                # Show progress at intervals or for the first/last items
                if i % progress_interval == 0 or i == total_examples - 1 or i < 3:
                    print(f"Evaluating example {i+1}/{total_examples} ({(i+1)/total_examples*100:.1f}%): '{ref[:50]}...' → '{hyp[:50]}...'")
                
                try:
                    # Call evaluate_pair
                    results = self.evaluate_pair(ref, hyp)
                    all_results.append(results)
                    print(f"Example {i+1} completed: Score={results.get('viseme_alignment_score', 'N/A'):.3f}")
                except Exception as e:
                    print(f"Error evaluating example {i+1}: {e}")
                    # Add minimal result to maintain example count
                    all_results.append({
                        'reference': ref,
                        'hypothesis': hyp,
                        'error': str(e),
                        'viseme_alignment_score': 0.0,
                        'phonetic_edit_distance': float('inf'),
                    })
            
            print(f"Completed evaluation of {len(all_results)} examples")
            
            # Only continue if we have results
            if not all_results:
                print("No results to analyze. Please check the evaluation process.")
                return None
                
            print("Generating visualizations and statistics...")
            
            # Save results first (in case visualization fails)
            results_file = os.path.join(output_dir, 'detailed_results.json')
            try:
                with open(results_file, 'w') as f:
                    # Convert numpy arrays to lists for JSON serialization
                    serializable_results = []
                    for res in all_results:
                        # Deep copy to avoid modifying the original
                        res_copy = dict(res)
                        # Handle non-serializable items
                        if 'phonetic_alignment' in res_copy:
                            res_copy['phonetic_alignment'] = [list(op) for op in res_copy['phonetic_alignment']]
                        serializable_results.append(res_copy)
                    
                    json.dump(serializable_results, f, indent=2)
                
                print(f"Saved all results to {results_file}")
            except Exception as e:
                print(f"Error saving results: {e}")
            
            # Try to generate visualizations
            try:
                # Analyze examples
                multi_visualizations = self.analyze_multiple_examples_with_results(all_results, output_dir)
                
                # Generate summary statistics
                summary = self.generate_dataset_summary(all_results, output_dir)
                
                return {
                    'summary': summary,
                    'visualizations': multi_visualizations,
                    'results_file': results_file,
                    'output_dir': output_dir
                }
            except Exception as e:
                print(f"Error generating visualizations: {e}")
                import traceback
                traceback.print_exc()
                return {
                    'results_file': results_file,
                    'output_dir': output_dir
                }
        
        except Exception as e:
            print(f"Error analyzing dataset: {e}")
            import traceback
            traceback.print_exc()
            return None

    def evaluate_pair(self, reference, hypothesis):
        """
        Evaluate a single reference-hypothesis pair for phonetic and viseme similarity
        
        Parameters:
        - reference: Reference text (ground truth)
        - hypothesis: Hypothesis text (predicted)
        
        Returns:
        - Dictionary with evaluation results
        """
        print(f"  Debug: Starting phoneme conversion for reference: '{reference[:30]}...'")
        start_time = time.time()
        
        # Convert texts to phoneme sequences
        try:
            ref_phonemes = self.text_to_phonemes(reference)
            print(f"  Debug: Reference phonemes converted in {time.time() - start_time:.2f}s")
            
            hyp_phonemes = self.text_to_phonemes(hypothesis)
            print(f"  Debug: Hypothesis phonemes converted in {time.time() - start_time:.2f}s")
            
            print(f"  Debug: Reference phonemes: {ref_phonemes[:10]}... ({len(ref_phonemes)} phonemes)")
            print(f"  Debug: Hypothesis phonemes: {hyp_phonemes[:10]}... ({len(hyp_phonemes)} phonemes)")
            
            # Calculate phonetic alignment and edit distance
            print(f"  Debug: Starting phonetic alignment...")
            align_start = time.time()
            alignment, edit_distance = self.calculate_phonetic_alignment(ref_phonemes, hyp_phonemes)
            print(f"  Debug: Phonetic alignment completed in {time.time() - align_start:.2f}s")
            
            # Convert phonemes to visemes
            print(f"  Debug: Converting to visemes...")
            ref_visemes = [self.phoneme_to_viseme.get(p, 'other') for p in ref_phonemes]
            hyp_visemes = [self.phoneme_to_viseme.get(p, 'other') for p in hyp_phonemes]
            
            # Calculate viseme-level alignment and score
            vis_start = time.time()
            viseme_alignment, viseme_edit_distance = self.calculate_viseme_alignment(ref_visemes, hyp_visemes)
            print(f"  Debug: Viseme alignment completed in {time.time() - vis_start:.2f}s")
            
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
                'viseme_alignment_score': viseme_score,
            }
            
            print(f"  Debug: Evaluation completed in {time.time() - start_time:.2f}s")
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
                'viseme_alignment_score': 0.0,
                'phonetic_edit_distance': float('inf'),
            }

    def calculate_phonetic_alignment(self, seq1, seq2, max_len=None):
        """
        Calculate alignment between two phoneme sequences using dynamic programming
        
        Parameters:
        - seq1: First phoneme sequence
        - seq2: Second phoneme sequence
        - max_len: Maximum sequence length to process (optional, for performance)
        
        Returns:
        - alignment: List of (operation, seq1_item, seq2_item) tuples
        - edit_distance: Edit distance score
        """
        # Only limit sequence length if max_len is explicitly provided
        if max_len is not None and (len(seq1) > max_len or len(seq2) > max_len):
            print(f"  Warning: Truncating long sequences ({len(seq1)}, {len(seq2)}) to {max_len}")
            seq1 = seq1[:max_len]
            seq2 = seq2[:max_len]
        
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
                # Calculate substitution cost based on phonetic similarity
                if seq1[i-1] == seq2[j-1]:
                    cost = 0  # Exact match
                else:
                    # Get phonetic similarity (0 to 1, where 0 is identical)
                    try:
                        similarity = self.calculate_phonetic_distance(seq1[i-1], seq2[j-1])
                    except Exception as e:
                        print(f"  Warning: Error calculating phonetic distance between '{seq1[i-1]}' and '{seq2[j-1]}': {e}")
                        similarity = 1.0  # Default to maximum distance
                    cost = similarity  # Use similarity as cost
                
                dp[i][j] = min(
                    dp[i-1][j] + 1,  # Deletion
                    dp[i][j-1] + 1,  # Insertion
                    dp[i-1][j-1] + cost  # Substitution
                )
        
        # Reconstruct alignment
        alignment = []
        i, j = m, n
        while i > 0 or j > 0:
            if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (0 if seq1[i-1] == seq2[j-1] else self.calculate_phonetic_distance(seq1[i-1], seq2[j-1])):
                # Substitution or match
                if seq1[i-1] == seq2[j-1]:
                    alignment.append(('match', seq1[i-1], seq2[j-1]))
                else:
                    alignment.append(('substitute', seq1[i-1], seq2[j-1]))
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
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
                if seq1[i-1] == seq2[j-1]:
                    cost = 0  # Exact match
                else:
                    cost = 1  # Different visemes
                
                dp[i][j] = min(
                    dp[i-1][j] + 1,  # Deletion
                    dp[i][j-1] + 1,  # Insertion
                    dp[i-1][j-1] + cost  # Substitution
                )
        
        # Reconstruct alignment
        alignment = []
        i, j = m, n
        while i > 0 or j > 0:
            if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (0 if seq1[i-1] == seq2[j-1] else 1):
                # Substitution or match
                if seq1[i-1] == seq2[j-1]:
                    alignment.append(('match', seq1[i-1], seq2[j-1]))
                else:
                    alignment.append(('substitute', seq1[i-1], seq2[j-1]))
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
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
    
    def text_to_phonemes(self, text):
        """Convert English text to phoneme sequence with caching"""
        # Check if we've already cached this text
        if hasattr(self, 'phoneme_cache') and text in self.phoneme_cache:
            return self.phoneme_cache[text]
            
        # Convert using G2P
        start_time = time.time()
        raw_phonemes = self.g2p(text)
        
        # Process phonemes to match our dictionary keys
        processed_phonemes = []
        i = 0
        while i < len(raw_phonemes):
            # Handle special cases for diphthongs and affricates
            if i < len(raw_phonemes) - 1:
                # Check for potential diphthongs
                dipth = raw_phonemes[i] + raw_phonemes[i+1]
                if dipth in self.phoneme_to_viseme:
                    processed_phonemes.append(dipth)
                    i += 2
                    continue
                
                # Check for affricates
                if raw_phonemes[i] == 'T' and raw_phonemes[i+1] == 'S':
                    processed_phonemes.append('tʃ')
                    i += 2
                    continue
                if raw_phonemes[i] == 'D' and raw_phonemes[i+1] == 'Z':
                    processed_phonemes.append('dʒ')
                    i += 2
                    continue
            
            # Map CMU phoneme format to IPA
            phoneme = raw_phonemes[i]
            # Map common CMU phonemes to IPA
            cmu_to_ipa = {
                'AA': 'ɑ', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔ', 'AW': 'aʊ',
                'AY': 'aɪ', 'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð',
                'EH': 'ɛ', 'ER': 'ɝ', 'EY': 'eɪ', 'F': 'f', 'G': 'g',
                'HH': 'h', 'IH': 'ɪ', 'IY': 'i', 'JH': 'dʒ', 'K': 'k',
                'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ', 'OW': 'oʊ',
                'OY': 'ɔɪ', 'P': 'p', 'R': 'ɹ', 'S': 's', 'SH': 'ʃ',
                'T': 't', 'TH': 'θ', 'UH': 'ʊ', 'UW': 'u', 'V': 'v',
                'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ'
            }
            
            if phoneme in cmu_to_ipa:
                processed_phonemes.append(cmu_to_ipa[phoneme])
            else:
                processed_phonemes.append(phoneme.lower())
            
            i += 1
            
        print(f"  Debug: G2P conversion took {time.time() - start_time:.2f}s for '{text[:30]}...'")
        
        # Cache the result to avoid recalculating
        if not hasattr(self, 'phoneme_cache'):
            self.phoneme_cache = {}
        self.phoneme_cache[text] = processed_phonemes
        
        return processed_phonemes
    
    def analyze_multiple_examples_with_results(self, all_results, output_dir):
        """
        Analyze patterns across multiple evaluation results
        
        Parameters:
        - all_results: List of result dictionaries from self.evaluate()
        - output_dir: Directory to save visualizations
        
        Returns:
        - Dictionary with visualization file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nSaving multi-example visualizations to: {output_dir}")
        
        # Dictionary to store visualization paths
        viz_files = {}
        
        # First, create comprehensive dataset analysis dashboard (new)
        viz_files['dataset_dashboard'] = self._create_dataset_analysis_dashboard(all_results, output_dir)
        
        # 1. Create aggregate viseme confusion matrix
        viz_files['agg_confusion'] = self._plot_aggregate_confusion_matrix(all_results, output_dir)
        
        # 2. Error rates by viseme class
        viz_files['viseme_error_rates'] = self._plot_viseme_error_rates(all_results, output_dir)
        
        # 3. Common substitution patterns
        viz_files['substitution_patterns'] = self._plot_substitution_patterns(all_results, output_dir)
        
        # 4. Performance by word length
        viz_files['word_length_performance'] = self._plot_word_length_performance(all_results, output_dir)
        
        # 5. Error distribution visualization
        viz_files['error_distribution'] = self._plot_error_distribution(all_results, output_dir)
        
        # 6. Word-level analysis - NEW
        viz_files['word_level_analysis'] = self._analyze_word_level_errors(all_results, output_dir)
        
        # 7. NEW: Phoneme confusion t-SNE visualization
        viz_files['phoneme_tsne'] = self._plot_phoneme_tsne(all_results, output_dir)
        
        # 8. NEW: Common error contexts
        viz_files['error_contexts'] = self._plot_error_contexts(all_results, output_dir)
        
        # 9. NEW: Viseme similarity heatmap
        viz_files['viseme_similarity'] = self._plot_viseme_similarity_heatmap(all_results, output_dir)
        
        # 10. NEW: Phoneme misrecognition rates
        viz_files['phoneme_error_rates'] = self._plot_phoneme_error_rates(all_results, output_dir)
        
        return viz_files
    
    def _create_dataset_analysis_dashboard(self, all_results, output_dir):
        """
        Create a comprehensive dashboard analyzing the entire dataset of references vs hypotheses
        
        Parameters:
        - all_results: List of result dictionaries with evaluation results
        - output_dir: Directory to save visualizations
        
        Returns:
        - Dictionary with paths to the generated visualizations
        """
        try:
            dashboard_files = {}
            
            # Create dashboard directory
            dashboard_dir = os.path.join(output_dir, 'dataset_dashboard')
            os.makedirs(dashboard_dir, exist_ok=True)
            
            # Extract references and hypotheses from results
            references = []
            hypotheses = []
            scores = []
            
            for result in all_results:
                if 'reference' in result and 'hypothesis' in result:
                    references.append(result['reference'])
                    hypotheses.append(result['hypothesis'])
                    if 'viseme_alignment_score' in result:
                        scores.append(result['viseme_alignment_score'])
            
            if not references or not hypotheses:
                print("No valid reference-hypothesis pairs found in results")
                return {}
            
            print(f"Creating dataset dashboard for {len(references)} examples")
            
            # 1. Basic Dataset Statistics Visualization
            plt.figure(figsize=(12, 10))
            
            # Calculate text length statistics
            ref_word_counts = [len(r.split()) for r in references]
            hyp_word_counts = [len(h.split()) for h in hypotheses]
            
            # Plot 1: Text length distribution
            plt.subplot(2, 2, 1)
            bins = np.arange(0, max(max(ref_word_counts), max(hyp_word_counts)) + 5, 5)
            plt.hist(ref_word_counts, bins=bins, alpha=0.7, label='Reference')
            plt.hist(hyp_word_counts, bins=bins, alpha=0.7, label='Hypothesis')
            plt.xlabel('Word Count')
            plt.ylabel('Frequency')
            plt.title('Distribution of Text Lengths')
            plt.legend()
            
            # Plot 2: Reference vs Hypothesis length scatter
            plt.subplot(2, 2, 2)
            plt.scatter(ref_word_counts, hyp_word_counts, alpha=0.5)
            max_count = max(max(ref_word_counts), max(hyp_word_counts)) + 1
            plt.plot([0, max_count], [0, max_count], 'r--')  # Diagonal line
            plt.xlabel('Reference Length (words)')
            plt.ylabel('Hypothesis Length (words)')
            plt.title('Text Length Comparison')
            
            # Plot 3: Score distribution
            plt.subplot(2, 2, 3)
            if scores:
                plt.hist(scores, bins=20, alpha=0.7, color='green')
                plt.axvline(np.mean(scores), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(scores):.3f}')
                plt.xlabel('Viseme Alignment Score')
                plt.ylabel('Frequency')
                plt.title('Score Distribution')
                plt.legend()
            else:
                plt.text(0.5, 0.5, "No scores available", ha='center', va='center')
                plt.title('Score Distribution')
            
            # Plot 4: Length vs Score relationship
            plt.subplot(2, 2, 4)
            if ref_word_counts and scores:
                plt.scatter(ref_word_counts, scores, alpha=0.5)
                plt.xlabel('Text Length (words)')
                plt.ylabel('Viseme Alignment Score')
                plt.title('Performance vs Text Length')
                
                # Add trend line if we have enough data
                if len(ref_word_counts) > 2:
                    try:
                        m, b = np.polyfit(ref_word_counts, scores, 1)
                        x = np.array([min(ref_word_counts), max(ref_word_counts)])
                        plt.plot(x, m*x + b, 'r--', label=f'Trend: y={m:.4f}x+{b:.4f}')
                        plt.legend()
                    except:
                        pass
            else:
                plt.text(0.5, 0.5, "Insufficient data", ha='center', va='center')
                plt.title('Performance vs Text Length')
            
            plt.tight_layout()
            stats_file = os.path.join(dashboard_dir, 'basic_statistics.png')
            plt.savefig(stats_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            dashboard_files['basic_stats'] = stats_file
            
            # 2. Word-level analysis: Top missing/added words
            # Analyze words present in reference but missing from hypothesis and vice versa
            missing_words = Counter()
            added_words = Counter()
            common_words = Counter()
            
            # Calculate word-level differences
            for ref, hyp in zip(references, hypotheses):
                ref_words = set(ref.lower().split())
                hyp_words = set(hyp.lower().split())
                
                # Words in reference but missing from hypothesis
                for word in ref_words - hyp_words:
                    missing_words[word] += 1
                
                # Words added in hypothesis (not in reference)
                for word in hyp_words - ref_words:
                    added_words[word] += 1
                
                # Words in both
                for word in ref_words & hyp_words:
                    common_words[word] += 1
            
            # Create visualization
            plt.figure(figsize=(15, 10))
            
            # Plot missing words
            plt.subplot(2, 1, 1)
            top_missing = missing_words.most_common(15)
            if top_missing:
                words, counts = zip(*top_missing)
                plt.barh(range(len(words)), counts, color='salmon')
                plt.yticks(range(len(words)), words)
                plt.xlabel('Frequency')
                plt.title('Top Words Missing in Hypothesis')
                plt.grid(axis='x', alpha=0.3)
            else:
                plt.text(0.5, 0.5, "No missing words found", ha='center', va='center')
                plt.title('Missing Words Analysis')
            
            # Plot added words
            plt.subplot(2, 1, 2)
            top_added = added_words.most_common(15)
            if top_added:
                words, counts = zip(*top_added)
                plt.barh(range(len(words)), counts, color='skyblue')
                plt.yticks(range(len(words)), words)
                plt.xlabel('Frequency')
                plt.title('Top Words Added in Hypothesis')
                plt.grid(axis='x', alpha=0.3)
            else:
                plt.text(0.5, 0.5, "No added words found", ha='center', va='center')
                plt.title('Added Words Analysis')
            
            plt.tight_layout()
            word_diff_file = os.path.join(dashboard_dir, 'word_differences.png')
            plt.savefig(word_diff_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            dashboard_files['word_diff'] = word_diff_file
            
            # 3. Character-level distribution comparison
            plt.figure(figsize=(14, 8))
            
            # Get character distributions
            ref_chars = Counter(''.join(references).lower())
            hyp_chars = Counter(''.join(hypotheses).lower())
            
            # Get all unique characters, filtered to common ones
            all_chars = sorted(set(ref_chars.keys()) | set(hyp_chars.keys()))
            common_chars = [c for c in all_chars if c.isalnum() or c in '.,-?! '][:25]
            
            # Plot character frequencies
            ref_counts = [ref_chars.get(c, 0) for c in common_chars]
            hyp_counts = [hyp_chars.get(c, 0) for c in common_chars]
            
            # Normalize to percentages
            ref_total = sum(ref_chars.values())
            hyp_total = sum(hyp_chars.values())
            
            if ref_total > 0 and hyp_total > 0:
                ref_pct = [count/ref_total*100 for count in ref_counts]
                hyp_pct = [count/hyp_total*100 for count in hyp_counts]
                
                x = np.arange(len(common_chars))
                width = 0.35
                
                plt.bar(x - width/2, ref_pct, width, label='Reference', color='royalblue')
                plt.bar(x + width/2, hyp_pct, width, label='Hypothesis', color='darkorange')
                
                plt.xlabel('Character')
                plt.ylabel('Percentage of Total (%)')
                plt.title('Character Distribution: Reference vs Hypothesis')
                plt.xticks(x, common_chars)
                plt.legend()
                plt.grid(axis='y', alpha=0.3)
            else:
                plt.text(0.5, 0.5, "Insufficient character data", ha='center', va='center')
                plt.title('Character Distribution')
            
            char_dist_file = os.path.join(dashboard_dir, 'character_distribution.png')
            plt.savefig(char_dist_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            dashboard_files['char_dist'] = char_dist_file
            
            # 4. Word error analysis
            # Calculate Word Error Rate (WER) for each example
            word_error_rates = []
            word_match_rates = []
            
            for ref, hyp in zip(references, hypotheses):
                ref_words = ref.lower().split()
                hyp_words = hyp.lower().split()
                
                if ref_words:
                    # Calculate operations using simple alignment
                    try:
                        alignment, edit_distance = self._calculate_word_alignment(ref_words, hyp_words)
                        word_error_rate = edit_distance / len(ref_words)
                        word_error_rates.append(word_error_rate)
                        
                        # Calculate simple word match rate
                        matches = sum(1 for op, _, _ in alignment if op == 'match')
                        word_match_rate = matches / len(ref_words) if ref_words else 0
                        word_match_rates.append(word_match_rate)
                    except Exception as e:
                        print(f"Error calculating word error rates: {e}")
            
            # Plot results
            plt.figure(figsize=(12, 6))
            
            # Plot word error rate distribution
            plt.subplot(1, 2, 1)
            if word_error_rates:
                plt.hist(word_error_rates, bins=np.linspace(0, min(max(word_error_rates), 2), 20), alpha=0.7)
                plt.axvline(np.mean(word_error_rates), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(word_error_rates):.3f}')
                plt.xlabel('Word Error Rate')
                plt.ylabel('Frequency')
                plt.title('Word Error Rate Distribution')
                plt.legend()
            else:
                plt.text(0.5, 0.5, "No word error rate data", ha='center', va='center')
                plt.title('Word Error Rate Distribution')
            
            # Plot word match rate distribution
            plt.subplot(1, 2, 2)
            if word_match_rates:
                plt.hist(word_match_rates, bins=np.linspace(0, 1, 20), alpha=0.7)
                plt.axvline(np.mean(word_match_rates), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(word_match_rates):.3f}')
                plt.xlabel('Word Match Rate')
                plt.ylabel('Frequency')
                plt.title('Word Match Rate Distribution')
                plt.legend()
            else:
                plt.text(0.5, 0.5, "No word match rate data", ha='center', va='center')
                plt.title('Word Match Rate Distribution')
            
            plt.tight_layout()
            word_error_file = os.path.join(dashboard_dir, 'word_error_analysis.png')
            plt.savefig(word_error_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            dashboard_files['word_error'] = word_error_file
            
            # 5. Example-by-example performance visualization
            plt.figure(figsize=(12, 6))
            
            # Get indices and scores
            indices = list(range(len(all_results)))
            
            # Plot viseme scores for each example
            if scores:
                plt.bar(indices, scores, alpha=0.7)
                avg_score = np.mean(scores)
                plt.axhline(avg_score, color='red', linestyle='--', 
                          label=f'Mean: {avg_score:.3f}')
                plt.xlabel('Example Index')
                plt.ylabel('Viseme Alignment Score')
                plt.title('Performance Across All Examples')
                plt.ylim(0, 1.0)
                plt.legend()
                plt.grid(alpha=0.3)
            else:
                plt.text(0.5, 0.5, "No score data available", ha='center', va='center')
                plt.title('Example Performance Visualization')
            
            performance_file = os.path.join(dashboard_dir, 'example_performance.png')
            plt.savefig(performance_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            dashboard_files['performance'] = performance_file
            
            # 6. Create a summary HTML file for easy viewing
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Dataset Analysis Dashboard</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2 {{ color: #333; }}
                    .viz-section {{ margin-bottom: 30px; }}
                    img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <h1>Dataset Analysis Dashboard</h1>
                <p>Analysis of {len(references)} reference-hypothesis pairs</p>
                
                <div class="viz-section">
                    <h2>1. Basic Dataset Statistics</h2>
                    <img src="basic_statistics.png" alt="Basic Statistics">
                </div>
                
                <div class="viz-section">
                    <h2>2. Word Differences Analysis</h2>
                    <img src="word_differences.png" alt="Word Differences">
                </div>
                
                <div class="viz-section">
                    <h2>3. Character Distribution Comparison</h2>
                    <img src="character_distribution.png" alt="Character Distribution">
                </div>
                
                <div class="viz-section">
                    <h2>4. Word Error Analysis</h2>
                    <img src="word_error_analysis.png" alt="Word Error Analysis">
                </div>
                
                <div class="viz-section">
                    <h2>5. Example-by-Example Performance</h2>
                    <img src="example_performance.png" alt="Example Performance">
                </div>
            </body>
            </html>
            """
            
            # Save HTML file
            html_file = os.path.join(dashboard_dir, 'dashboard.html')
            with open(html_file, 'w') as f:
                f.write(html_content)
            
            dashboard_files['html'] = html_file
            
            # 7. Save a text summary with key statistics
            # Calculate overall metrics
            if scores:
                avg_score = np.mean(scores)
                min_score = min(scores)
                max_score = max(scores)
            else:
                avg_score = min_score = max_score = "N/A"
            
            # Word-level metrics
            if word_error_rates:
                avg_wer = np.mean(word_error_rates)
                avg_match = np.mean(word_match_rates)
            else:
                avg_wer = avg_match = "N/A"
            
            # Text length metrics
            avg_ref_len = np.mean(ref_word_counts) if ref_word_counts else "N/A"
            avg_hyp_len = np.mean(hyp_word_counts) if hyp_word_counts else "N/A"
            
            # Missing/added word stats
            top_5_missing = missing_words.most_common(5)
            top_5_added = added_words.most_common(5)
            
            summary_text = f"""
            DATASET ANALYSIS SUMMARY
            =======================
            
            Number of examples: {len(references)}
            
            Performance Metrics:
            - Average viseme alignment score: {avg_score if isinstance(avg_score, str) else f"{avg_score:.3f}"}
            - Min score: {min_score if isinstance(min_score, str) else f"{min_score:.3f}"}
            - Max score: {max_score if isinstance(max_score, str) else f"{max_score:.3f}"}
            - Average word error rate: {avg_wer if isinstance(avg_wer, str) else f"{avg_wer:.3f}"}
            - Average word match rate: {avg_match if isinstance(avg_match, str) else f"{avg_match:.3f}"}
            
            Text Length Stats:
            - Average reference length: {avg_ref_len if isinstance(avg_ref_len, str) else f"{avg_ref_len:.1f}"} words
            - Average hypothesis length: {avg_hyp_len if isinstance(avg_hyp_len, str) else f"{avg_hyp_len:.1f}"} words
            
            Top 5 Most Frequently Missing Words:
            {", ".join([f"'{word}' ({count})" for word, count in top_5_missing]) if top_5_missing else "None"}
            
            Top 5 Most Frequently Added Words:
            {", ".join([f"'{word}' ({count})" for word, count in top_5_added]) if top_5_added else "None"}
            """
            
            # Save text summary
            text_file = os.path.join(dashboard_dir, 'summary.txt')
            with open(text_file, 'w') as f:
                f.write(summary_text)
            
            dashboard_files['text'] = text_file
            
            print(f"Dataset analysis dashboard created at: {dashboard_dir}")
            print(f"Open {html_file} to view all visualizations.")
            
            return dashboard_files
            
        except Exception as e:
            print(f"Error creating dataset analysis dashboard: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _plot_aggregate_confusion_matrix(self, all_results, output_dir):
        """
        Create an aggregate confusion matrix across all examples
        
        Parameters:
        - all_results: List of result dictionaries from self.evaluate()
        - output_dir: Directory to save visualization
        
        Returns:
        - Path to saved visualization
        """
        # Extract all viseme substitutions
        viseme_subs = []
        for result in all_results:
            for op in result.get('phonetic_alignment', []):
                if op[0] == 'substitute':
                    ref_phoneme = op[1]
                    hyp_phoneme = op[2]
                    
                    # Get viseme classes
                    ref_viseme = self.phoneme_to_viseme.get(ref_phoneme, 'other')
                    hyp_viseme = self.phoneme_to_viseme.get(hyp_phoneme, 'other')
                    
                    viseme_subs.append((ref_viseme, hyp_viseme))
        
        # Get all unique viseme classes
        all_visemes = sorted(set(self.phoneme_to_viseme.values()))
        
        # Create confusion matrix
        n_visemes = len(all_visemes)
        confusion_matrix = np.zeros((n_visemes, n_visemes))
        
        # Index mapping for viseme classes
        viseme_to_index = {viseme: i for i, viseme in enumerate(all_visemes)}
        
        # Populate confusion matrix
        for ref_viseme, hyp_viseme in viseme_subs:
            if ref_viseme in viseme_to_index and hyp_viseme in viseme_to_index:
                ref_idx = viseme_to_index[ref_viseme]
                hyp_idx = viseme_to_index[hyp_viseme]
                confusion_matrix[ref_idx, hyp_idx] += 1
        
        # Normalize by row (reference) sums
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        norm_confusion = np.zeros_like(confusion_matrix)
        
        for i in range(n_visemes):
            if row_sums[i] > 0:
                norm_confusion[i, :] = confusion_matrix[i, :] / row_sums[i]
        
        # Create visualization
        plt.figure(figsize=(14, 12))
        
        # Define custom colormap from white to dark blue
        cmap = LinearSegmentedColormap.from_list('custom_cmap', ['white', '#003b6f'])
        
        # Plot confusion matrix
        sns.heatmap(norm_confusion, annot=True, fmt=".2f", cmap=cmap,
                   xticklabels=all_visemes, yticklabels=all_visemes, 
                   cbar_kws={'label': 'Normalized Frequency'})
        
        plt.title('Viseme Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted Viseme', fontsize=14)
        plt.ylabel('Reference Viseme', fontsize=14)
        
        # Rotate x labels for better visibility
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(output_dir, 'viseme_confusion_matrix.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def _plot_viseme_error_rates(self, all_results, output_dir):
        """
        Create visualization of error rates by viseme class
        
        Parameters:
        - all_results: List of result dictionaries from self.evaluate()
        - output_dir: Directory to save visualization
        
        Returns:
        - Path to saved visualization
        """
        # Count errors by viseme class
        viseme_counts = {}
        viseme_errors = {}
        
        for result in all_results:
            for op in result.get('phonetic_alignment', []):
                if op[0] == 'match' or op[0] == 'substitute':
                    ref_phoneme = op[1]
                    ref_viseme = self.phoneme_to_viseme.get(ref_phoneme, 'other')
                    
                    # Increment total count for this viseme
                    if ref_viseme not in viseme_counts:
                        viseme_counts[ref_viseme] = 0
                        viseme_errors[ref_viseme] = 0
                    
                    viseme_counts[ref_viseme] += 1
                    
                    # If substitution, increment error count
                    if op[0] == 'substitute':
                        viseme_errors[ref_viseme] += 1
        
        # Calculate error rates
        viseme_error_rates = {}
        for viseme in viseme_counts:
            if viseme_counts[viseme] > 0:
                viseme_error_rates[viseme] = viseme_errors[viseme] / viseme_counts[viseme]
            else:
                viseme_error_rates[viseme] = 0
        
        # Sort by error rate
        sorted_visemes = sorted(viseme_error_rates.keys(), 
                               key=lambda v: viseme_error_rates[v], reverse=True)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Extract data for plotting
        visemes = [v for v in sorted_visemes if viseme_counts[v] >= 5]  # Filter out rare visemes
        error_rates = [viseme_error_rates[v] for v in visemes]
        counts = [viseme_counts[v] for v in visemes]
        
        # Plot error rates as bars
        bars = plt.bar(visemes, error_rates, alpha=0.7)
        
        # Add count labels above bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'n={count}', ha='center', va='bottom', fontsize=9)
        
        plt.title('Error Rate by Viseme Class', fontsize=16)
        plt.xlabel('Viseme Class', fontsize=14)
        plt.ylabel('Error Rate', fontsize=14)
        plt.ylim(0, min(1.0, max(error_rates) * 1.2))  # Set y limit with headroom
        
        # Rotate x labels for better visibility
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(output_dir, 'viseme_error_rates.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def _plot_substitution_patterns(self, all_results, output_dir):
        """
        Create network visualization of common substitution patterns
        
        Parameters:
        - all_results: List of result dictionaries from self.evaluate()
        - output_dir: Directory to save visualization
        
        Returns:
        - Path to saved visualization
        """
        # Extract all phoneme substitutions
        phoneme_subs = []
        for result in all_results:
            for op in result.get('phonetic_alignment', []):
                if op[0] == 'substitute':
                    ref_phoneme = op[1]
                    hyp_phoneme = op[2]
                    phoneme_subs.append((ref_phoneme, hyp_phoneme))
        
        # Count substitution frequencies
        sub_counts = Counter(phoneme_subs)
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes and edges
        min_count = 2  # Minimum count to include
        max_count = max(sub_counts.values()) if sub_counts else 1
        
        for (ref, hyp), count in sub_counts.items():
            if count >= min_count:
                # Add nodes with viseme class information
                ref_viseme = self.phoneme_to_viseme.get(ref, 'other')
                hyp_viseme = self.phoneme_to_viseme.get(hyp, 'other')
                
                G.add_node(ref, viseme=ref_viseme)
                G.add_node(hyp, viseme=hyp_viseme)
                
                # Add edge with count information
                G.add_edge(ref, hyp, weight=count, width=1 + 4 * (count / max_count))
        
        # Create visualization if we have edges
        if G.number_of_edges() > 0:
            plt.figure(figsize=(14, 12))
            
            # Set up node positions using spring layout
            pos = nx.spring_layout(G, k=0.3, iterations=50)
            
            # Create viseme-based color map
            viseme_classes = sorted(set(nx.get_node_attributes(G, 'viseme').values()))
            color_map = plt.cm.get_cmap('tab20', len(viseme_classes))
            viseme_colors = {v: color_map(i) for i, v in enumerate(viseme_classes)}
            
            # Get node colors based on viseme class
            node_colors = [viseme_colors[G.nodes[n]['viseme']] for n in G.nodes]
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors, alpha=0.8)
            
            # Draw edges with varying width based on count
            edge_widths = [G[u][v]['width'] for u, v in G.edges]
            nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, 
                                  edge_color='gray', arrowsize=15)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')
            
            # Add edge labels (counts)
            edge_labels = {(u, v): f"{G[u][v]['weight']}" for u, v in G.edges}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
            
            # Add legend for viseme classes
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor=viseme_colors[v], markersize=10, 
                                         label=v) for v in viseme_classes]
            plt.legend(handles=legend_elements, title="Viseme Classes", 
                      loc='upper left', bbox_to_anchor=(1.05, 1))
            
            plt.title('Common Phoneme Substitution Patterns', fontsize=16)
            plt.axis('off')
            plt.tight_layout()
            
            # Save figure
            output_file = os.path.join(output_dir, 'substitution_patterns.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_file
        else:
            return None
    
    def _plot_word_length_performance(self, all_results, output_dir):
        """
        Create visualization of performance by word length
        
        Parameters:
        - all_results: List of result dictionaries from self.evaluate()
        - output_dir: Directory to save visualization
        
        Returns:
        - Path to saved visualization
        """
        # Group examples by length of reference text
        length_to_scores = {}
        
        for result in all_results:
            if 'reference' in result and 'viseme_alignment_score' in result:
                # Use word count as length
                word_count = len(result['reference'].split())
                
                if word_count not in length_to_scores:
                    length_to_scores[word_count] = []
                
                length_to_scores[word_count].append(result['viseme_alignment_score'])
        
        # Calculate mean and std for each length
        lengths = sorted(length_to_scores.keys())
        mean_scores = [np.mean(length_to_scores[l]) for l in lengths]
        std_scores = [np.std(length_to_scores[l]) for l in lengths]
        counts = [len(length_to_scores[l]) for l in lengths]
        
        # Create visualization
        plt.figure(figsize=(12, 7))
        
        # Plot mean scores with error bars
        plt.errorbar(lengths, mean_scores, yerr=std_scores, fmt='o-', 
                    capsize=5, elinewidth=1, markersize=8)
        
        # Add count labels
        for i, (x, y, count) in enumerate(zip(lengths, mean_scores, counts)):
            plt.annotate(f'n={count}', (x, y), xytext=(0, 10), 
                        textcoords='offset points', ha='center')
        
        plt.title('Performance by Word Length', fontsize=16)
        plt.xlabel('Number of Words', fontsize=14)
        plt.ylabel('Viseme Alignment Score', fontsize=14)
        plt.grid(linestyle='--', alpha=0.7)
        
        # Format x-axis to show integer ticks
        plt.xticks(lengths)
        
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(output_dir, 'performance_by_length.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def _plot_error_distribution(self, all_results, output_dir):
        """
        Create visualization of error distribution
        
        Parameters:
        - all_results: List of result dictionaries from self.evaluate()
        - output_dir: Directory to save visualization
        
        Returns:
        - Path to saved visualization
        """
        # Extract scores
        viseme_scores = [r['viseme_alignment_score'] for r in all_results]
        phonetic_distances = [r['phonetic_edit_distance'] for r in all_results]
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot viseme alignment score distribution
        sns.histplot(viseme_scores, kde=True, ax=ax1, color='cornflowerblue')
        ax1.set_title('Viseme Alignment Score Distribution', fontsize=14)
        ax1.set_xlabel('Viseme Alignment Score', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.axvline(np.mean(viseme_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(viseme_scores):.3f}')
        ax1.legend()
        
        # Plot phonetic edit distance distribution
        sns.histplot(phonetic_distances, kde=True, ax=ax2, color='salmon')
        ax2.set_title('Phonetic Edit Distance Distribution', fontsize=14)
        ax2.set_xlabel('Phonetic Edit Distance', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.axvline(np.mean(phonetic_distances), color='blue', linestyle='--',
                   label=f'Mean: {np.mean(phonetic_distances):.3f}')
        ax2.legend()
        
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(output_dir, 'error_distribution.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_file

    def _plot_phoneme_tsne(self, all_results, output_dir):
        """
        Create t-SNE visualization of phoneme similarity based on confusion patterns
        
        Parameters:
        - all_results: List of result dictionaries
        - output_dir: Directory to save visualization
        
        Returns:
        - Path to saved visualization
        """
        try:
            # Extract all phoneme substitutions
            phoneme_subs = []
            all_phonemes = set()
            
            for result in all_results:
                for op in result.get('phonetic_alignment', []):
                    if op[0] == 'substitute':
                        ref_phoneme = op[1]
                        hyp_phoneme = op[2]
                        phoneme_subs.append((ref_phoneme, hyp_phoneme))
                        all_phonemes.add(ref_phoneme)
                        all_phonemes.add(hyp_phoneme)
            
            # Only proceed if we have enough data
            if len(all_phonemes) < 5 or len(phoneme_subs) < 10:
                print("Not enough phoneme substitution data for t-SNE visualization")
                return None
            
            # Create co-occurrence matrix for substitutions
            phonemes_list = sorted(list(all_phonemes))
            phoneme_to_idx = {p: i for i, p in enumerate(phonemes_list)}
            
            # Initialize co-occurrence matrix
            cooc_matrix = np.zeros((len(phonemes_list), len(phonemes_list)))
            
            # Fill co-occurrence matrix
            for ref, hyp in phoneme_subs:
                ref_idx = phoneme_to_idx[ref]
                hyp_idx = phoneme_to_idx[hyp]
                cooc_matrix[ref_idx, hyp_idx] += 1
                cooc_matrix[hyp_idx, ref_idx] += 1  # Make symmetric
            
            # Add diagonal values (self-similarity)
            np.fill_diagonal(cooc_matrix, cooc_matrix.max())
            
            # Normalize matrix
            row_sums = cooc_matrix.sum(axis=1).reshape(-1, 1)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            norm_matrix = cooc_matrix / row_sums
            
            # Apply t-SNE
            tsne = TSNE(n_components=2, perplexity=min(5, len(phonemes_list)-1), 
                       random_state=42, learning_rate=200)
            tsne_results = tsne.fit_transform(norm_matrix)
            
            # Create visualization
            plt.figure(figsize=(12, 10))
            
            # Get viseme classes for coloring
            phoneme_visemes = [self.phoneme_to_viseme.get(p, 'other') for p in phonemes_list]
            viseme_classes = sorted(list(set(phoneme_visemes)))
            
            # Create colormap
            cmap = plt.cm.get_cmap('tab20', len(viseme_classes))
            viseme_to_color = {v: cmap(i) for i, v in enumerate(viseme_classes)}
            point_colors = [viseme_to_color[v] for v in phoneme_visemes]
            
            # Plot points
            plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=point_colors, alpha=0.7, s=100)
            
            # Add labels
            for i, phoneme in enumerate(phonemes_list):
                plt.annotate(phoneme, (tsne_results[i, 0], tsne_results[i, 1]), 
                            fontsize=10, ha='center', va='center')
            
            # Add legend for viseme classes
            handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=viseme_to_color[v], 
                                 markersize=10, label=v) for v in viseme_classes]
            plt.legend(handles=handles, title="Viseme Classes", 
                      loc='best', frameon=True, framealpha=0.8)
            
            plt.title("Phoneme Similarity Space (t-SNE)", fontsize=16)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            
            # Save figure
            output_file = os.path.join(output_dir, 'phoneme_tsne.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_file
        except Exception as e:
            print(f"Error creating t-SNE visualization: {e}")
            return None

    def _plot_error_contexts(self, all_results, output_dir):
        """
        Analyze and visualize common error contexts (what phonemes often appear before/after errors)
        
        Parameters:
        - all_results: List of result dictionaries
        - output_dir: Directory to save visualization
        
        Returns:
        - Path to saved visualization
        """
        try:
            # Extract contexts around errors
            pre_error_contexts = []
            post_error_contexts = []
            
            for result in all_results:
                alignment = result.get('phonetic_alignment', [])
                for i, op in enumerate(alignment):
                    # Look for substitutions, insertions, deletions
                    if op[0] != 'match':
                        # Get pre-context (1 phoneme before error)
                        if i > 0 and alignment[i-1][0] == 'match':
                            pre_error_contexts.append(alignment[i-1][1])
                        
                        # Get post-context (1 phoneme after error)
                        if i < len(alignment) - 1 and alignment[i+1][0] == 'match':
                            post_error_contexts.append(alignment[i+1][1])
            
            # Only proceed if we have enough data
            if len(pre_error_contexts) < 5 or len(post_error_contexts) < 5:
                print("Not enough context data for error context visualization")
                return None
            
            # Count frequencies
            pre_counter = Counter(pre_error_contexts)
            post_counter = Counter(post_error_contexts)
            
            # Get most common contexts
            top_n = 15
            most_common_pre = pre_counter.most_common(top_n)
            most_common_post = post_counter.most_common(top_n)
        
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Plot pre-error contexts
            pre_labels = [p for p, _ in most_common_pre]
            pre_counts = [c for _, c in most_common_pre]
            pre_visemes = [self.phoneme_to_viseme.get(p, 'other') for p in pre_labels]
            pre_colors = plt.cm.tab10([hash(v) % 10 for v in pre_visemes])
            
            ax1.barh(range(len(pre_labels)), pre_counts, color=pre_colors)
            ax1.set_yticks(range(len(pre_labels)))
            ax1.set_yticklabels([f"{p} ({v})" for p, v in zip(pre_labels, pre_visemes)])
            ax1.set_title("Phonemes Before Errors", fontsize=14)
            ax1.set_xlabel("Count", fontsize=12)
            ax1.grid(alpha=0.3, axis='x')
            
            # Plot post-error contexts
            post_labels = [p for p, _ in most_common_post]
            post_counts = [c for _, c in most_common_post]
            post_visemes = [self.phoneme_to_viseme.get(p, 'other') for p in post_labels]
            post_colors = plt.cm.tab10([hash(v) % 10 for v in post_visemes])
            
            ax2.barh(range(len(post_labels)), post_counts, color=post_colors)
            ax2.set_yticks(range(len(post_labels)))
            ax2.set_yticklabels([f"{p} ({v})" for p, v in zip(post_labels, post_visemes)])
            ax2.set_title("Phonemes After Errors", fontsize=14)
            ax2.set_xlabel("Count", fontsize=12)
            ax2.grid(alpha=0.3, axis='x')
            
            plt.tight_layout()
            
            # Save figure
            output_file = os.path.join(output_dir, 'error_contexts.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_file
        except Exception as e:
            print(f"Error creating error contexts visualization: {e}")
            return None

    def _plot_viseme_similarity_heatmap(self, all_results, output_dir):
        """
        Create a heatmap showing the similarity between viseme classes
        
        Parameters:
        - all_results: List of result dictionaries
        - output_dir: Directory to save visualization
        
        Returns:
        - Path to saved visualization
        """
        try:
            # Get all viseme classes
            viseme_classes = sorted(set(self.phoneme_to_viseme.values()))
            n_visemes = len(viseme_classes)
            
            # Create indices for viseme classes
            viseme_to_idx = {v: i for i, v in enumerate(viseme_classes)}
            
            # Initialize similarity matrix
            sim_matrix = np.zeros((n_visemes, n_visemes))
            
            # Fill similarity matrix based on feature similarity or substitution patterns
            
            # Method 1: Based on phoneme features
            for v1 in viseme_classes:
                for v2 in viseme_classes:
                    # Get phonemes for each viseme
                    v1_phonemes = [p for p, v in self.phoneme_to_viseme.items() if v == v1]
                    v2_phonemes = [p for p, v in self.phoneme_to_viseme.items() if v == v2]
                    
                    # Skip if no phonemes found
                    if not v1_phonemes or not v2_phonemes:
                        continue
                    
                    # Calculate average distance between all phoneme pairs
                    total_dist = 0
                    count = 0
                    
                    for p1 in v1_phonemes:
                        for p2 in v2_phonemes:
                            try:
                                if p1 in self.phoneme_features and p2 in self.phoneme_features:
                                    dist = self.calculate_phonetic_distance(p1, p2)
                                    total_dist += dist
                                    count += 1
                            except:
                                pass
                    
                    # Compute average distance and convert to similarity
                    if count > 0:
                        avg_dist = total_dist / count
                        # Convert distance to similarity (1 - distance)
                        sim_matrix[viseme_to_idx[v1], viseme_to_idx[v2]] = 1.0 - avg_dist
            
            # Create visualization
            plt.figure(figsize=(12, 10))
            
            # Create custom colormap from white to dark blue
            cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#f7fbff', '#08306b'])
            
            # Plot similarity matrix
            sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap=cmap,
                       xticklabels=viseme_classes, yticklabels=viseme_classes, 
                       cbar_kws={'label': 'Similarity'})
            
            plt.title('Viseme Similarity Heatmap', fontsize=16)
            plt.xlabel('Viseme Class', fontsize=14)
            plt.ylabel('Viseme Class', fontsize=14)
            
            # Rotate x labels for better visibility
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save figure
            output_file = os.path.join(output_dir, 'viseme_similarity_heatmap.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_file
        except Exception as e:
            print(f"Error creating viseme similarity heatmap: {e}")
            return None

    def _plot_phoneme_error_rates(self, all_results, output_dir):
        """
        Visualize error rates for individual phonemes
        
        Parameters:
        - all_results: List of result dictionaries
        - output_dir: Directory to save visualization
        
        Returns:
        - Path to saved visualization
        """
        try:
            # Count occurrences and errors for each phoneme
            phoneme_counts = {}
            phoneme_errors = {}
            
            for result in all_results:
                alignment = result.get('phonetic_alignment', [])
                
                for op in alignment:
                    if op[0] == 'match' or op[0] == 'substitute':
                        # Reference phoneme
                        ref_phoneme = op[1]
                        if ref_phoneme not in phoneme_counts:
                            phoneme_counts[ref_phoneme] = 0
                            phoneme_errors[ref_phoneme] = 0
                        
                        phoneme_counts[ref_phoneme] += 1
                        
                        # Count errors (substitutions)
                        if op[0] == 'substitute':
                            phoneme_errors[ref_phoneme] += 1
            
            # Calculate error rates
            error_rates = {}
            for phoneme in phoneme_counts:
                if phoneme_counts[phoneme] >= 5:  # Only include phonemes with sufficient samples
                    error_rates[phoneme] = phoneme_errors[phoneme] / phoneme_counts[phoneme]
            
            # Sort phonemes by error rate
            sorted_phonemes = sorted(error_rates.keys(), key=lambda p: error_rates[p], reverse=True)
            
            # Limit to top N for readability
            top_n = 30
            top_phonemes = sorted_phonemes[:top_n]
        
            # Create figure
            plt.figure(figsize=(14, 10))
            
            # Extract data for plotting
            labels = top_phonemes
            rates = [error_rates[p] for p in top_phonemes]
            counts = [phoneme_counts[p] for p in top_phonemes]
            visemes = [self.phoneme_to_viseme.get(p, 'other') for p in top_phonemes]
            
            # Create color mapping based on viseme classes
            unique_visemes = sorted(set(visemes))
            viseme_to_color = {v: plt.cm.tab20(i % 20) for i, v in enumerate(unique_visemes)}
            colors = [viseme_to_color[v] for v in visemes]
            
            # Plot error rates
            bars = plt.bar(range(len(labels)), rates, color=colors, alpha=0.7)
            
            # Add count annotations
            for i, (bar, count) in enumerate(zip(bars, counts)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f"{count}", ha='center', va='bottom', fontsize=9)
            
            # Add viseme labels
            for i, (label, viseme) in enumerate(zip(labels, visemes)):
                plt.text(i, -0.03, f"{viseme}", rotation=45, ha='right', fontsize=8, alpha=0.7)
            
            # Formatting
            plt.xticks(range(len(labels)), labels)
            plt.title("Phoneme Error Rates", fontsize=16)
            plt.ylabel("Error Rate", fontsize=14)
            plt.xlabel("Phoneme (with Viseme Class below)", fontsize=14)
            plt.ylim(0, min(1.0, max(rates) * 1.2))
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add legend for viseme classes
            legend_elements = [Patch(facecolor=viseme_to_color[v], label=v) 
                              for v in unique_visemes]
            plt.legend(handles=legend_elements, title="Viseme Classes", 
                      loc='upper right')
            
            plt.tight_layout()
            
            # Save figure
            output_file = os.path.join(output_dir, 'phoneme_error_rates.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_file
        except Exception as e:
            print(f"Error creating phoneme error rates visualization: {e}")
            return None

    def analyze_multiple_examples(self, example_pairs, output_dir=None):
        """
        Analyze patterns across multiple reference-hypothesis pairs
        
        Parameters:
        - example_pairs: List of (reference, hypothesis) tuples
        - output_dir: Directory to save visualizations (default: create timestamped dir)
        
        Returns:
        - Dictionary with visualization file paths
        """
        # Create output directory if not provided
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                     "plots", f"multi_analysis_{timestamp}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nAnalyzing {len(example_pairs)} examples...")
        
        # Process all examples to get evaluation results
        all_results = []
        for i, (ref, hyp) in enumerate(example_pairs):
            print(f"Evaluating example {i+1}/{len(example_pairs)}: '{ref}' → '{hyp}'")
            results = self.evaluate_pair(ref, hyp)
            all_results.append(results)
        
        # Call the analysis method that works with results
        return self.analyze_multiple_examples_with_results(all_results, output_dir)

    def generate_dataset_summary(self, all_results, output_dir):
        """Generate summary statistics for a dataset of results"""
        summary = {
            'num_examples': len(all_results),
            'avg_viseme_score': np.mean([r['viseme_alignment_score'] for r in all_results if 'viseme_alignment_score' in r]),
            'avg_phonetic_distance': np.mean([r['phonetic_edit_distance'] for r in all_results if 'phonetic_edit_distance' in r and not np.isinf(r['phonetic_edit_distance'])]),
            'viseme_score_std': np.std([r['viseme_alignment_score'] for r in all_results if 'viseme_alignment_score' in r]),
            'phonetic_distance_std': np.std([r['phonetic_edit_distance'] for r in all_results if 'phonetic_edit_distance' in r and not np.isinf(r['phonetic_edit_distance'])]),
        }
        
        # Count operations across all examples
        op_counts = {"match": 0, "substitute": 0, "delete": 0, "insert": 0}
        for result in all_results:
            if 'phonetic_alignment' in result:
                for op in result['phonetic_alignment']:
                    op_counts[op[0]] += 1
        
        summary['operations'] = op_counts
        total_ops = sum(op_counts.values())
        summary['operations_percent'] = {k: (v / total_ops) * 100 for k, v in op_counts.items()} if total_ops > 0 else {k: 0 for k in op_counts}
        
        # Save summary to file
        summary_file = os.path.join(output_dir, 'summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Also create a human-readable text summary
        text_summary = os.path.join(output_dir, 'summary.txt')
        with open(text_summary, 'w') as f:
            f.write("=== Lip Reading Evaluation Dataset Summary ===\n\n")
            f.write(f"Number of examples: {summary['num_examples']}\n\n")
            f.write("Performance Metrics:\n")
            f.write(f"- Average Viseme Alignment Score: {summary['avg_viseme_score']:.3f} ± {summary['viseme_score_std']:.3f} (higher is better)\n")
            f.write(f"- Average Phonetic Edit Distance: {summary['avg_phonetic_distance']:.3f} ± {summary['phonetic_distance_std']:.3f} (lower is better)\n\n")
            f.write("Operation Counts:\n")
            for op, count in op_counts.items():
                f.write(f"- {op.capitalize()}: {count} ({summary['operations_percent'][op]:.1f}%)\n")
        
        return summary

    def calculate_phonetic_distance(self, phoneme1, phoneme2):
        """
        Calculate phonetic distance between two phonemes using their feature vectors
        with a more generalized approach based on linguistic features.
        
        This uses a weighted feature-based distance metric that takes into account
        the importance of different features for visual speech perception.
        """
        # Check cache to avoid recalculating
        cache_key = (phoneme1, phoneme2)
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        # Reversed key check
        reversed_key = (phoneme2, phoneme1)
        if reversed_key in self.similarity_cache:
            return self.similarity_cache[reversed_key]
        
        # Special case: if either is not in our phoneme inventory
        if phoneme1 not in self.phoneme_features or phoneme2 not in self.phoneme_features:
            # Default to maximum distance
            return 1.0
        
        # Try to use panphon's sophisticated distance function first
        try:
            dst = panphon.distance.Distance()
            # Calculate weighted feature edit distance
            dist = dst.feature_edit_distance(phoneme1, phoneme2)
            # Normalize to 0-1 range (panphon distances can be larger)
            norm_dist = min(1.0, dist / 10.0)
            # Cache and return
            self.similarity_cache[cache_key] = norm_dist
            return norm_dist
        except:
            # Fall back to our custom feature-based distance
            pass
        
        # Get feature dictionaries
        features1 = self.phoneme_features[phoneme1]
        features2 = self.phoneme_features[phoneme2]
        
        # Collect all features from both phonemes
        all_features = set(features1.keys()) | set(features2.keys())
        
        # Calculate weighted distance
        total_distance = 0.0
        total_weight = 0.0
        
        for feature in all_features:
            # Skip features missing weight information
            if feature not in self.feature_weights:
                continue
                
            weight = self.feature_weights[feature]
            
            # Get feature values (default to None if missing)
            value1 = features1.get(feature, None)
            value2 = features2.get(feature, None)
            
            # Skip if either phoneme doesn't have this feature
            if value1 is None or value2 is None:
                continue
                
            # Calculate feature-specific distance
            feature_dist = 0.0
            if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
                # Numeric features - normalize by typical range
                if feature == 'place':
                    # Place of articulation (1-7 scale)
                    feature_dist = abs(value1 - value2) / 6.0
                elif feature == 'height' or feature == 'backness':
                    # Vowel features (typically 1-4 or 1-3 scales)
                    feature_dist = abs(value1 - value2) / 3.0
                else:
                    # Binary or other numeric features
                    feature_dist = abs(value1 - value2)
            elif value1 == value2:
                # Matching categorical features
                feature_dist = 0.0
            else:
                # Non-matching categorical features
                feature_dist = 1.0
            
            # Add weighted contribution to total distance
            total_distance += weight * feature_dist
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            normalized_distance = total_distance / total_weight
        else:
            # If no weighted features were compared, use viseme comparison
            # Same viseme = more similar
            normalized_distance = 0.0 if self.phoneme_to_viseme.get(phoneme1) == self.phoneme_to_viseme.get(phoneme2) else 1.0
        
        # Cache result for future lookups
        self.similarity_cache[cache_key] = normalized_distance
        
        return normalized_distance

    def _calculate_word_alignment(self, seq1, seq2):
        """
        Calculate alignment between two word sequences using dynamic programming
        
        Parameters:
        - seq1: First word sequence
        - seq2: Second word sequence
        
        Returns:
        - alignment: List of (operation, seq1_idx, seq2_idx) tuples
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
                # Simple cost model: 0 for match, 1 for mismatch
                cost = 0 if seq1[i-1].lower() == seq2[j-1].lower() else 1
                
                dp[i][j] = min(
                    dp[i-1][j] + 1,  # Deletion
                    dp[i][j-1] + 1,  # Insertion
                    dp[i-1][j-1] + cost  # Substitution
                )
        
        # Reconstruct alignment
        alignment = []
        i, j = m, n
        
        while i > 0 or j > 0:
            if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (0 if seq1[i-1].lower() == seq2[j-1].lower() else 1):
                # Substitution or match
                if seq1[i-1].lower() == seq2[j-1].lower():
                    alignment.append(('match', i-1, j-1))
                else:
                    alignment.append(('substitute', i-1, j-1))
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
                # Deletion
                alignment.append(('delete', i-1, -1))
                i -= 1
            else:
                # Insertion
                alignment.append(('insert', -1, j-1))
                j -= 1
        
        # Reverse alignment to get correct order
        alignment.reverse()
        
        return alignment, dp[m][n]

    def _analyze_word_level_errors(self, all_results, output_dir):
        """
        Analyze word-level errors and patterns across examples
        
        Parameters:
        - all_results: List of result dictionaries
        - output_dir: Directory to save visualizations
        
        Returns:
        - Dictionary with paths to generated visualizations
        """
        try:
            # Create output dictionary
            word_viz_files = {}
            
            # Extract word pairs from all examples
            references = []
            hypotheses = []
            
            for result in all_results:
                if 'reference' in result and 'hypothesis' in result:
                    references.append(result['reference'])
                    hypotheses.append(result['hypothesis'])
            
            if not references or not hypotheses:
                return {}
            
            # Create word pairs directory
            word_dir = os.path.join(output_dir, 'word_analysis')
            os.makedirs(word_dir, exist_ok=True)
            
            # Find common word substitutions
            word_substitutions = []
            
            for ref, hyp in zip(references, hypotheses):
                ref_words = ref.lower().split()
                hyp_words = hyp.lower().split()
                
                # Get alignment
                alignment, _ = self._calculate_word_alignment(ref_words, hyp_words)
                
                # Extract substitutions
                for op, ref_idx, hyp_idx in alignment:
                    if op == 'substitute':
                        ref_word = ref_words[ref_idx]
                        hyp_word = hyp_words[hyp_idx]
                        word_substitutions.append((ref_word, hyp_word))
            
            # Count frequencies
            sub_counts = Counter(word_substitutions)
            
            # Get top substitutions
            top_n = 15
            most_common = sub_counts.most_common(top_n)
            
            # Create visualization if we have data
            if most_common:
                plt.figure(figsize=(12, 8))
                
                # Prepare data
                labels = [f"'{ref}' → '{hyp}'" for (ref, hyp), _ in most_common]
                counts = [count for _, count in most_common]
                
                # Create horizontal bar chart
                plt.barh(range(len(labels)), counts, color='skyblue')
                plt.yticks(range(len(labels)), labels)
                plt.xlabel('Frequency')
                plt.title('Most Common Word Substitutions')
                plt.grid(axis='x', alpha=0.3)
                
                # Save figure
                sub_file = os.path.join(word_dir, 'common_word_substitutions.png')
                plt.savefig(sub_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                word_viz_files['substitutions'] = sub_file
            
            # Create word error rate by length visualization
            word_lengths = {}
            word_errors = {}
            
            for ref, hyp in zip(references, hypotheses):
                ref_words = ref.lower().split()
                hyp_words = hyp.lower().split()
                
                # Get alignment
                alignment, _ = self._calculate_word_alignment(ref_words, hyp_words)
                
                # Analyze errors by word length
                for op, ref_idx, hyp_idx in alignment:
                    if op != 'match' and ref_idx >= 0 and ref_idx < len(ref_words):
                        # Word error
                        word = ref_words[ref_idx]
                        length = len(word)
                        
                        if length not in word_errors:
                            word_errors[length] = 0
                            word_lengths[length] = 0
                        
                        word_errors[length] += 1
                    
                    if ref_idx >= 0 and ref_idx < len(ref_words):
                        # Count total words by length
                        word = ref_words[ref_idx]
                        length = len(word)
                        
                        if length not in word_lengths:
                            word_lengths[length] = 0
                        
                        word_lengths[length] += 1
            
            # Create visualization if we have data
            if word_lengths:
                plt.figure(figsize=(10, 6))
                
                # Calculate error rates
                lengths = sorted(word_lengths.keys())
                error_rates = []
                
                for length in lengths:
                    total = word_lengths.get(length, 0)
                    errors = word_errors.get(length, 0)
                    rate = errors / total if total > 0 else 0
                    error_rates.append(rate)
                
                # Create bar chart with data counts
                bars = plt.bar(lengths, error_rates, alpha=0.7)
                
                # Add count annotations
                for i, length in enumerate(lengths):
                    plt.text(lengths[i], error_rates[i] + 0.01, 
                            f'n={word_lengths[length]}', ha='center', va='bottom', fontsize=8)
                
                plt.xlabel('Word Length (characters)')
                plt.ylabel('Error Rate')
                plt.title('Word Error Rate by Length')
                plt.grid(axis='y', alpha=0.3)
                plt.ylim(0, min(1.0, max(error_rates) * 1.2) if error_rates else 1.0)
                
                # Save figure
                length_file = os.path.join(word_dir, 'error_rate_by_length.png')
                plt.savefig(length_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                word_viz_files['length'] = length_file
            
            return word_viz_files
            
        except Exception as e:
            print(f"Error in word-level analysis: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _calculate_word_alignment(self, seq1, seq2):
        """
        Calculate alignment between two word sequences
        
        Parameters:
        - seq1: First word sequence (list of words)
        - seq2: Second word sequence (list of words)
        
        Returns:
        - alignment: List of (operation, seq1_idx, seq2_idx) tuples
        - edit_distance: Total edit distance
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
                if seq1[i-1].lower() == seq2[j-1].lower():
                    dp[i][j] = dp[i-1][j-1]  # Match
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],    # Deletion
                        dp[i][j-1],    # Insertion
                        dp[i-1][j-1]   # Substitution
                    )
        
        # Reconstruct alignment
        alignment = []
        i, j = m, n
        
        while i > 0 or j > 0:
            if i > 0 and j > 0 and seq1[i-1].lower() == seq2[j-1].lower():
                # Match
                alignment.append(('match', i-1, j-1))
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
                # Substitution
                alignment.append(('substitute', i-1, j-1))
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
                # Deletion
                alignment.append(('delete', i-1, -1))
                i -= 1
            else:
                # Insertion
                alignment.append(('insert', -1, j-1))
                j -= 1
        
        # Reverse alignment
        alignment.reverse()
        
        return alignment, dp[m][n]

# Update main function to include JSON file processing
def main():
    evaluator = LipReadingEvaluator()
    
    # Example usage with JSON file
    import argparse
    parser = argparse.ArgumentParser(description='Lip Reading Evaluation Tool')
    parser.add_argument('--json', type=str, help='JSON file with reference-hypothesis pairs')
    parser.add_argument('--output', type=str, help='Output directory for visualizations', default=None)
    parser.add_argument('--features', type=str, help='Custom phoneme features JSON file (to load)', default=None)
    parser.add_argument('--save-features', type=str, help='Save phoneme features to JSON file', default=None)
    parser.add_argument('--max-samples', type=int, help='Maximum number of samples to process', default=None)
    parser.add_argument('--example', action='store_true', help='Run with example data')
    args = parser.parse_args()
    
    # Load custom phoneme features if specified
    if args.features:
        evaluator = LipReadingEvaluator(load_from_file=args.features)
    
    # Save phoneme features if requested
    if args.save_features:
        evaluator.save_phoneme_features(args.save_features)
    
    # Process JSON file if provided
    if args.json:
        results = evaluator.analyze_json_dataset(args.json, args.output, args.max_samples)
        if results:
            print(f"\nAnalysis complete. Output saved to: {results['output_dir']}")
    
    # Run with example data if requested
    elif args.example:
        # Example of analyzing multiple pairs together
        example_pairs = [
            ("holy sites", "hallways"),
            ("nine eleven", "9 11"),
            ("the doctor prescribed medication", "the doctor described medication"),
            ("please pay careful attention", "please play careful attention"),
            ("the bright sunshine warmed my face", "the bright sunshine warmed my faith"),
            ("I'm reading your lips", "I'm leading your lips"),
            ("speaking clearly helps lip reading", "speaking nearly helps lip reading"),
            ("bat", "pat"),
            ("mail", "nail"),
            ("key", "tea"),
            ("weather forecast", "whether forecast"),
            ("the meeting will be short", "the meaning will be short")
        ]
        
        # Analyze all examples individually
        for ref, hyp in example_pairs[:3]:  # First few examples
            results = evaluator.evaluate_pair(ref, hyp)
            print(f"Results for '{ref}' → '{hyp}':")
            print(f"- Viseme Alignment Score: {results['viseme_alignment_score']:.3f}")
            print(f"- Phonetic Edit Distance: {results['phonetic_edit_distance']:.3f}")
        
        # Analyze all examples together
        multi_visualizations = evaluator.analyze_multiple_examples(example_pairs)
        
        print("\nMulti-example analysis visualizations:")
        for viz_type, file_path in multi_visualizations.items():
            print(f"- {viz_type}: {file_path}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()