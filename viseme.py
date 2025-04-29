import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.patches import Patch
from collections import Counter, defaultdict
from itertools import chain
from datetime import datetime
from g2p_en import G2p  # Grapheme-to-phoneme converter
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


class LipReadingEvaluator:
    """Comprehensive lip reading evaluation system with information-theoretic weights"""
    
    def __init__(self, load_from_file=None):
        """Initialize the evaluator with phoneme mappings and converters"""
        
        # Import essential libraries
        try:
            import g2p_en
            self.g2p = g2p_en.G2p()
        except ImportError:
            print("ERROR: g2p_en library is required for phoneme conversion")
            print("Install with: pip install g2p_en")
            raise
            
        try:
            import panphon
            self.ft = panphon.featuretable.FeatureTable()
            self.dst = panphon.distance.Distance()
            self._has_panphon = True
        except ImportError:
            print("ERROR: panphon library is required for phonetic features")
            print("Install with: pip install panphon")
            raise
        
        # Define phoneme-to-viseme mapping using Microsoft's 22-category system
        self.phoneme_to_viseme = {
            # 0: Silence
            '.': 0, ' ': 0, '-': 0, 
            
            # 1: Open/central vowels (æ, ə, ʌ)
            'æ': 1, 'ə': 1, 'ʌ': 1,
            
            # 2: Open back vowels
            'ɑ': 2, 'ɒ': 2,
            
            # 3: Open-mid back rounded
            'ɔ': 3,
            
            # 4: Mid vowels
            'ɛ': 4, 'ʊ': 4, 'e': 4,
            
            # 5: R-colored vowels
            'ɝ': 5, 'ɚ': 5,
            
            # 6: Close front vowels + /j/
            'i': 6, 'ɪ': 6, 'j': 6,
            
            # 7: Close back rounded + /w/
            'u': 7, 'w': 7,
            
            # 8: Close-mid back rounded
            'o': 8,
            
            # 9-11: Major diphthongs
            'aʊ': 9, 'ɔɪ': 10, 'aɪ': 11,
            'eɪ': 6, 'oʊ': 8, 'ɪə': 6,
            'eə': 4, 'ʊə': 4,
            
            # 12: Glottal
            'h': 12, 'ʔ': 12,
            
            # 13: Rhotic approximant
            'ɹ': 13, 'r': 13, 'ɾ': 13,
            
            # 14: Lateral approximant
            'l': 14,
            
            # 15: Alveolar fricatives
            's': 15, 'z': 15,
            
            # 16: Post-alveolar sounds
            'ʃ': 16, 'ʒ': 16, 'tʃ': 16, 'dʒ': 16,
            
            # 17: Voiced dental fricative
            'ð': 17,
            
            # 18: Labiodental fricatives
            'f': 18, 'v': 18,
            
            # 19: Alveolar stops, nasal + voiceless dental
            't': 19, 'd': 19, 'n': 19, 'θ': 19,
            
            # 20: Velar consonants
            'k': 20, 'g': 20, 'ŋ': 20, 'ɲ': 20,
            
            # 21: Bilabial consonants
            'p': 21, 'b': 21, 'm': 21
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
        
        # Initialize phoneme feature cache
        self.phoneme_features_cache = {}
        
        # Pre-calculate phoneme similarity matrix based on panphon distance
        self.similarity_cache = {}

    def get_phoneme_features(self, phoneme):
        """
        Get the phonetic features for a given phoneme using panphon.
        
        Args:
            phoneme: The phoneme to get features for
            
        Returns:
            dict: A dictionary of phonetic features
        """
        # Check if we've cached this phoneme's features
        if phoneme in self.phoneme_features_cache:
            return self.phoneme_features_cache[phoneme]
            
        # Handle special characters
        if phoneme in ['.', ' ', '-'] or not phoneme.strip():
            features = {'is_silence': 1}
            self.phoneme_features_cache[phoneme] = features
            return features
            
        # Try to get panphon features
        try:
            # Get feature vector from panphon
            feature_vector = self.ft.word_to_vector_list(phoneme, numeric=True)
            
            if not feature_vector:
                # If panphon couldn't process it, create minimal features
                features = {'unknown': 1}
                self.phoneme_features_cache[phoneme] = features
                return features
                
            # Convert to dictionary with feature names
            feature_dict = {}
            for i, feature_name in enumerate(self.ft.names):
                feature_dict[feature_name] = int(feature_vector[0][i])
                
            # Add derived features for convenience
            feature_dict['is_vowel'] = feature_dict.get('syl', 0) == 1
            feature_dict['is_consonant'] = feature_dict.get('syl', 0) == 0
            
            # Cache and return
            self.phoneme_features_cache[phoneme] = feature_dict
            return feature_dict
            
        except Exception as e:
            print(f"Warning: Error getting features for '{phoneme}': {e}")
            # Create minimal features if panphon fails
            features = {'error': 1}
            self.phoneme_features_cache[phoneme] = features
            return features

    def calculate_phonetic_distance(self, phoneme1, phoneme2):
        """
        Calculate phonetic distance between two phonemes using panphon's distance metrics.
        
        Args:
            phoneme1: First phoneme
            phoneme2: Second phoneme
            
        Returns:
            float: Distance value between 0.0 (identical) and 1.0 (maximally different)
        """
        # Check cache to avoid recalculating
        cache_key = (phoneme1, phoneme2)
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        # Reversed key check
        reversed_key = (phoneme2, phoneme1)
        if reversed_key in self.similarity_cache:
            return self.similarity_cache[reversed_key]
        
        # Special case: if either is silence or special character
        if phoneme1 in ['.', ' ', '-'] or phoneme2 in ['.', ' ', '-']:
            # Silence compared to anything is maximally different
            distance = 1.0
            self.similarity_cache[cache_key] = distance
            return distance
            
        # Calculate distance using panphon
        try:
            # Calculate weighted feature edit distance
            distance = self.dst.feature_edit_distance(phoneme1, phoneme2)
            
            # Normalize to 0-1 range (panphon distances can be larger)
            norm_distance = min(1.0, distance / 10.0)
            
            # Cache and return
            self.similarity_cache[cache_key] = norm_distance
            return norm_distance
            
        except Exception as e:
            print(f"Warning: Error calculating distance between '{phoneme1}' and '{phoneme2}': {e}")
            # Default to maximum distance on error
            self.similarity_cache[cache_key] = 1.0
            return 1.0
    
    def calculate_information_theoretic_weights(self):
        """
        Calculate feature weights based on information theory principles
        using entropy and visual distinctiveness.
        """
        weights = {}
        phoneme_inventory = list(self.phoneme_features_cache.keys())
        
        # For each feature type (place, manner, voiced, etc.)
        all_features = set()
        for features in self.phoneme_features_cache.values():
            all_features.update(features.keys())
        
        # Calculate entropy-based weights for each feature
        feature_entropies = {}
        for feature in all_features:
            # Get all values this feature takes across the phoneme inventory
            feature_values = []
            for phoneme in phoneme_inventory:
                if feature in self.phoneme_features_cache[phoneme]:
                    feature_values.append(self.phoneme_features_cache[phoneme][feature])
            
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
                if feature not in self.phoneme_features_cache[phoneme]:
                    continue
                    
                feature_value = self.phoneme_features_cache[phoneme][feature]
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
            
            # Try to generate visualizations
            try:
                # Analyze examples
                multi_visualizations = self.analyze_multiple_examples_with_results(all_results, output_dir)
                
                # Generate summary statistics
                summary = self.generate_dataset_summary(all_results, output_dir)
                
                return {
                    'summary': summary,
                    'visualizations': multi_visualizations,
                    'output_dir': output_dir
                }
            except Exception as e:
                print(f"Error generating visualizations: {e}")
                import traceback
                traceback.print_exc()
                return {
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
            
            # Convert phonemes to visemes using most appropriate mapping
            print(f"  Debug: Converting to visemes...")
            ref_visemes = [self.get_closest_viseme(p) for p in ref_phonemes]
            hyp_visemes = [self.get_closest_viseme(p) for p in hyp_phonemes]
            
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
                # Strip stress markers from both phonemes before checking diphthongs
                base_p1 = raw_phonemes[i].rstrip('0123456789')
                base_p2 = raw_phonemes[i+1].rstrip('0123456789')
                
                # Check for potential diphthongs
                dipth = base_p1 + base_p2
                if dipth in self.phoneme_to_viseme:
                    processed_phonemes.append(dipth)
                    i += 2
                    continue
                
                # Check for affricates
                if base_p1 == 'T' and base_p2 == 'S':
                    processed_phonemes.append('tʃ')
                    i += 2
                    continue
                if base_p1 == 'D' and base_p2 == 'Z':
                    processed_phonemes.append('dʒ')
                    i += 2
                    continue
            
            # Map CMU phoneme format to IPA
            phoneme = raw_phonemes[i]
            
            # Strip stress markers (numbers) before mapping
            base_phoneme = phoneme.rstrip('0123456789')
            
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
            
            if base_phoneme in cmu_to_ipa:
                processed_phonemes.append(cmu_to_ipa[base_phoneme])
            else:
                # For phonemes not in our mapping, use lowercase without stress markers
                processed_phonemes.append(base_phoneme.lower())
            
            i += 1
            
        print(f"  Debug: G2P conversion took {time.time() - start_time:.2f}s for '{text[:30]}...'")
        
        # Cache the result to avoid recalculating
        if not hasattr(self, 'phoneme_cache'):
            self.phoneme_cache = {}
        self.phoneme_cache[text] = processed_phonemes
        
        return processed_phonemes
    
    def analyze_multiple_examples_with_results(self, all_results, output_dir):
        """
        Analyze multiple examples with their existing evaluation results
        
        Parameters:
        - all_results: List of result dictionaries with evaluation results
        - output_dir: Directory to save visualizations
        
        Returns:
        - Dictionary with paths to the generated visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nSaving multi-example visualizations to: {output_dir}")
        
        # Dictionary to store visualization paths
        viz_files = {}
        
        # 1. Create aggregate viseme confusion matrix
        viz_files['agg_confusion'] = self._plot_aggregate_confusion_matrix(all_results, output_dir)
        
        # 2. Error rates by viseme class
        viz_files['viseme_error_rates'] = self._plot_viseme_error_rates(all_results, output_dir)
        
        # 3. Substitution patterns (REMOVED - causes errors and not useful)
        # viz_files['substitution_patterns'] = self._plot_substitution_patterns(all_results, output_dir)
        
        # 4. Word length vs performance
        viz_files['word_length'] = self._plot_word_length_performance(all_results, output_dir)
        
        # 5. Error distribution
        viz_files['error_distribution'] = self._plot_error_distribution(all_results, output_dir)
        
        # 6. Phoneme t-SNE (REMOVED - causes errors and not useful)
        # viz_files['phoneme_tsne'] = self._plot_phoneme_tsne(all_results, output_dir)
        
        # 7. Error contexts
        viz_files['error_contexts'] = self._plot_error_contexts(all_results, output_dir)
        
        # 8. Viseme similarity heatmap
        viz_files['viseme_similarity'] = self._plot_viseme_similarity_heatmap(all_results, output_dir)
        
        # 9. Individual phoneme error rates
        viz_files['phoneme_errors'] = self._plot_phoneme_error_rates(all_results, output_dir)
        
        # Generate dataset summary
        viz_files['summary'] = self.generate_dataset_summary(all_results, output_dir)
        
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
                    
                    # Get viseme classes (using closest viseme mapping)
                    ref_viseme = self.get_closest_viseme(ref_phoneme)
                    hyp_viseme = self.get_closest_viseme(hyp_phoneme)
                    
                    viseme_subs.append((ref_viseme, hyp_viseme))
        
        # Get all unique viseme classes (should only be 0-21)
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
        
        # Define custom colormap from white to dark blue using the newer API
        cmap = LinearSegmentedColormap.from_list('custom_cmap', ['white', '#003b6f'])
        
        # Create viseme labels using the viseme_id_to_name mapping for readability
        viseme_labels = [f"{v}: {self.viseme_id_to_name[v]}" for v in all_visemes]
        
        # Plot confusion matrix
        sns.heatmap(norm_confusion, annot=True, fmt=".2f", cmap=cmap,
                   xticklabels=viseme_labels, yticklabels=viseme_labels, 
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
                    ref_viseme = self.get_closest_viseme(ref_phoneme)
                    
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
        
        # Create viseme labels using the viseme_id_to_name mapping for readability
        viseme_labels = [f"{v}: {self.viseme_id_to_name[v]}" for v in visemes]
        
        # Plot error rates as bars
        bars = plt.bar(viseme_labels, error_rates, alpha=0.7)
        
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
        Visualize patterns of viseme substitutions as a heatmap
        
        Parameters:
        - all_results: List of result dictionaries
        - output_dir: Directory to save visualization
        
        Returns:
        - Path to saved visualization
        """
        try:
            # Get all viseme substitutions from all results
            substitutions = []
            for result in all_results:
                for viseme_sub in result.get('viseme_substitutions', []):
                    ref_viseme = viseme_sub[0]
                    hyp_viseme = viseme_sub[1]
                    substitutions.append((ref_viseme, hyp_viseme))
            
            if not substitutions:
                print("No substitutions found to visualize")
                return None
        
            # Count substitution frequencies
            substitution_counts = {}
            for ref, hyp in substitutions:
                if (ref, hyp) not in substitution_counts:
                    substitution_counts[(ref, hyp)] = 0
                substitution_counts[(ref, hyp)] += 1
            
            # Get unique viseme classes
            unique_visemes = set()
            for ref, hyp in substitutions:
                unique_visemes.add(ref)
                unique_visemes.add(hyp)
            
            # Sort viseme classes for consistency
            unique_visemes = sorted(unique_visemes)
            
            # Create mapping of viseme to index
            viseme_to_index = {v: i for i, v in enumerate(unique_visemes)}
            
            # Create empty matrix
            n = len(unique_visemes)
            matrix = np.zeros((n, n))
            
            # Fill matrix with substitution counts
            for (ref, hyp), count in substitution_counts.items():
                ref_idx = viseme_to_index[ref]
                hyp_idx = viseme_to_index[hyp]
                matrix[ref_idx, hyp_idx] = count
            
            # Create viseme labels for better readability
            viseme_labels = [f"{v}: {self.viseme_id_to_name[v]}" for v in unique_visemes]
            
            # Create figure
            plt.figure(figsize=(14, 12))
            
            # Create heatmap with log scale for better visualization
            norm = LogNorm(vmin=1, vmax=max(1, np.max(matrix)))
            im = plt.imshow(matrix, norm=norm, cmap='viridis')
            
            # Add colorbar
            cbar = plt.colorbar(im)
            cbar.set_label('Substitution Count (log scale)')
            
            # Configure tick labels
            plt.xticks(range(n), viseme_labels, rotation=90)
            plt.yticks(range(n), viseme_labels)
            
            plt.xlabel('Hypothesis Viseme')
            plt.ylabel('Reference Viseme')
            plt.title('Viseme Substitution Patterns', fontsize=16)
            
            # Add text annotations for counts
            for i in range(n):
                for j in range(n):
                    count = matrix[i, j]
                    if count >= 3:  # Only show significant counts
                        plt.text(j, i, int(count), ha='center', va='center', 
                                color='white' if count > 10 else 'black',
                                fontsize=8)
            
            plt.tight_layout()
            
            # Save figure
            output_file = os.path.join(output_dir, 'substitution_patterns.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_file
        except Exception as e:
            print(f"Error creating substitution patterns visualization: {e}")
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
        Create t-SNE visualization of phoneme relationships
        
        Parameters:
        - all_results: List of result dictionaries
        - output_dir: Directory to save visualization
        
        Returns:
        - Path to saved visualization
        """
        try:
            # Skip this visualization as it doesn't work well with numeric viseme IDs
            print("Skipping t-SNE visualization as it's not compatible with numeric viseme IDs")
            return None
            
            # Rest of the function...
        except Exception as e:
            print(f"Error creating t-SNE visualization: {e}")
            return None

    def _plot_error_contexts(self, all_results, output_dir):
        """
        Create visualization of phoneme contexts around errors
        
        Parameters:
        - all_results: List of result dictionaries
        - output_dir: Directory to save visualization
        
        Returns:
        - Path to saved visualization
        """
        try:
            # Extract all errors (substitutions)
            error_contexts = []
            
            for result in all_results:
                alignment = result.get('phonetic_alignment', [])
                
                # Loop through alignment operations
                for i, op in enumerate(alignment):
                    if op[0] == 'substitute':
                        # Find context (previous and next phonemes)
                        pre_idx = max(0, i-2)
                        post_idx = min(len(alignment)-1, i+2)
                        
                        pre_phonemes = []
                        post_phonemes = []
                        
                        # Get preceding phonemes
                        for j in range(pre_idx, i):
                            if alignment[j][0] == 'match' or alignment[j][0] == 'substitute':
                                pre_phonemes.append(alignment[j][1])
                        
                        # Get following phonemes
                        for j in range(i+1, post_idx+1):
                            if j < len(alignment) and (alignment[j][0] == 'match' or alignment[j][0] == 'substitute'):
                                post_phonemes.append(alignment[j][1])
                        
                        # Add context to list
                        error_contexts.append({
                            'phoneme': op[1],
                            'pre': pre_phonemes,
                            'post': post_phonemes
                        })
            
            # Count frequency of phonemes before and after errors
            pre_counts = Counter()
            post_counts = Counter()
            
            for ctx in error_contexts:
                for p in ctx['pre']:
                    pre_counts[p] += 1
                for p in ctx['post']:
                    post_counts[p] += 1
            
            # Get most common phonemes in each position
            top_n = 15
            pre_top = [p for p, _ in pre_counts.most_common(top_n)]
            post_top = [p for p, _ in post_counts.most_common(top_n)]
            
            # Convert to viseme classes for coloring
            pre_labels = pre_top
            pre_visemes = [self.phoneme_to_viseme.get(p, -1) for p in pre_labels]  # Use -1 for 'other'
            
            post_labels = post_top
            post_visemes = [self.phoneme_to_viseme.get(p, -1) for p in post_labels]  # Use -1 for 'other'
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
            
            # Get counts for plotting
            pre_values = [pre_counts[p] for p in pre_labels]
            post_values = [post_counts[p] for p in post_labels]
            
            # Create viseme-based color mapping using the newer API
            viseme_classes = sorted(set(pre_visemes + post_visemes))
            cmap = plt.colormaps['tab20']
            
            # Create color arrays
            pre_colors = [cmap(viseme_classes.index(v) % 20) for v in pre_visemes]
            post_colors = [cmap(viseme_classes.index(v) % 20) for v in post_visemes]
            
            # Plot horizontal bar charts
            ax1.barh(range(len(pre_labels)), pre_values, color=pre_colors, alpha=0.7)
            ax2.barh(range(len(post_labels)), post_values, color=post_colors, alpha=0.7)
            
            # Add viseme info to labels
            pre_labels_with_viseme = [f"{p} ({v})" for p, v in zip(pre_labels, pre_visemes)]
            post_labels_with_viseme = [f"{p} ({v})" for p, v in zip(post_labels, post_visemes)]
            
            # Set labels
            ax1.set_yticks(range(len(pre_labels)))
            ax1.set_yticklabels(pre_labels_with_viseme)
            ax1.set_title("Phonemes Before Errors", fontsize=14)
            ax1.set_xlabel("Count", fontsize=12)
            ax1.grid(alpha=0.3, axis='x')
            
            ax2.set_yticks(range(len(post_labels)))
            ax2.set_yticklabels(post_labels_with_viseme)
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
            
            # Add perfect similarity along diagonal (each viseme is identical to itself)
            for i in range(n_visemes):
                sim_matrix[i, i] = 1.0
            
            # Enhance similarity matrix with data from substitutions in results
            substitution_counts = {}
            total_counts = {}
            
            # Count viseme substitutions from results
            for result in all_results:
                for op, idx1, idx2 in result.get('viseme_alignment', []):
                    if op == 'substitute':
                        # Add to substitution counter
                        if (idx1, idx2) not in substitution_counts:
                            substitution_counts[(idx1, idx2)] = 0
                        substitution_counts[(idx1, idx2)] += 1
                        
                        # Track total occurrences of first viseme
                        if idx1 not in total_counts:
                            total_counts[idx1] = 0
                        total_counts[idx1] += 1
            
            # Use substitution patterns to enhance similarity matrix
            # (more frequent substitutions = more similar visemes)
            for (v1, v2), count in substitution_counts.items():
                if v1 in viseme_to_idx and v2 in viseme_to_idx and v1 in total_counts:
                    # Calculate similarity based on substitution frequency
                    # (normalized by total occurrences of the first viseme)
                    i, j = viseme_to_idx[v1], viseme_to_idx[v2]
                    similarity = count / total_counts[v1]
                    
                    # Update similarity matrix (capped at 0.8 to preserve diagonal dominance)
                    sim_matrix[i, j] = min(0.8, similarity)
                    sim_matrix[j, i] = min(0.8, similarity)  # Make it symmetric
            
            # Fill in remaining similarities based on phonetic features
            for i, v1 in enumerate(viseme_classes):
                for j, v2 in enumerate(viseme_classes):
                    # Skip if already set from substitutions or diagonal
                    if sim_matrix[i, j] > 0:
                        continue
                        
                    # Get phonemes for each viseme
                    v1_phonemes = [p for p, v in self.phoneme_to_viseme.items() if v == v1]
                    v2_phonemes = [p for p, v in self.phoneme_to_viseme.items() if v == v2]
                    
                    # Skip if no phonemes found
                    if not v1_phonemes or not v2_phonemes:
                        continue
                    
                    # Calculate average distance between sample phoneme pairs
                    # (limit to a few samples for efficiency)
                    sample_p1 = v1_phonemes[:3]  # Take up to 3 phonemes from each viseme
                    sample_p2 = v2_phonemes[:3]
                    
                    total_dist = 0
                    count = 0
                    
                    for p1 in sample_p1:
                        for p2 in sample_p2:
                            try:
                                    dist = self.calculate_phonetic_distance(p1, p2)
                                    total_dist += dist
                                    count += 1
                            except Exception as e:
                                pass
                    
                    # Compute average distance and convert to similarity
                    if count > 0:
                        avg_dist = total_dist / count
                        # Convert distance to similarity (1 - distance)
                        # Cap at 0.7 to keep distinction from substitution-based similarity
                        sim_matrix[i, j] = min(0.7, 1.0 - avg_dist)
            
            # Apply contextual adjustment - similar viseme groups should have higher similarity
            viseme_groups = {
                'vowels': [1, 2, 3, 4, 5, 6, 7, 8],        # All vowels
                'diphthongs': [9, 10, 11],                 # All diphthongs
                'front_consonants': [17, 18, 19, 21],      # Front-articulated consonants
                'back_consonants': [12, 13, 14, 15, 16, 20] # Back-articulated consonants
            }
            
            # Boost similarity within groups
            for group_name, group_members in viseme_groups.items():
                for v1 in group_members:
                    for v2 in group_members:
                        if v1 != v2 and v1 in viseme_to_idx and v2 in viseme_to_idx:
                            i, j = viseme_to_idx[v1], viseme_to_idx[v2]
                            # Boost similarity by 0.2 but keep below 0.9
                            sim_matrix[i, j] = min(0.9, sim_matrix[i, j] + 0.2)
            
            # Create visualization
            plt.figure(figsize=(14, 12))
            
            # Create custom colormap from white to dark blue
            cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#f7fbff', '#08306b'])
            
            # Create viseme labels for the plot
            viseme_labels = []
            for v in viseme_classes:
                if v in self.viseme_id_to_name:
                    viseme_labels.append(f"{v}: {self.viseme_id_to_name[v]}")
                else:
                    viseme_labels.append(f"Other ({v})")
            
            # Plot similarity matrix
            sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap=cmap,
                       xticklabels=viseme_labels, yticklabels=viseme_labels, 
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
            import traceback
            traceback.print_exc()
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
            visemes = [self.get_closest_viseme(p) for p in top_phonemes]
            
            # Create viseme labels for better readability
            viseme_labels = [f"{v}: {self.viseme_id_to_name[v]}" for v in visemes]
            
            # Create color mapping based on viseme classes
            unique_visemes = sorted(set(visemes))
            cmap = plt.colormaps['tab20'] 
            viseme_to_color = {v: cmap(i % 20) for i, v in enumerate(unique_visemes)}
            colors = [viseme_to_color[v] for v in visemes]
            
            # Plot error rates
            bars = plt.bar(range(len(labels)), rates, color=colors, alpha=0.7)
            
            # Add count annotations
            for i, (bar, count) in enumerate(zip(bars, counts)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f"{count}", ha='center', va='bottom', fontsize=9)
            
            # Add viseme labels
            for i, (label, viseme_label) in enumerate(zip(labels, viseme_labels)):
                plt.text(i, -0.03, f"{viseme_label}", rotation=45, ha='right', fontsize=8, alpha=0.7)
            
            # Formatting
            plt.xticks(range(len(labels)), labels)
            plt.title("Phoneme Error Rates", fontsize=16)
            plt.ylabel("Error Rate", fontsize=14)
            plt.xlabel("Phoneme (with Viseme Class below)", fontsize=14)
            plt.ylim(0, min(1.0, max(rates) * 1.2))
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add legend for viseme classes
            legend_elements = [Patch(facecolor=viseme_to_color[v], 
                                    label=f"{v}: {self.viseme_id_to_name[v]}") 
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
        
        # Calculate Character Error Rate (CER)
        total_chars = 0
        char_errors = 0
        for result in all_results:
            if 'reference' in result and 'hypothesis' in result:
                ref = result['reference']
                hyp = result['hypothesis']
                total_chars += len(ref)
                # Calculate character-level edit distance
                char_edit_distance = nltk.edit_distance(ref, hyp)
                char_errors += char_edit_distance
        
        cer = char_errors / total_chars if total_chars > 0 else 0
        summary['cer'] = cer
        
        # Calculate Word Error Rate (WER)
        total_words = 0
        word_errors = 0
        for result in all_results:
            if 'reference' in result and 'hypothesis' in result:
                ref_words = result['reference'].split()
                hyp_words = result['hypothesis'].split()
                total_words += len(ref_words)
                # Calculate word-level edit distance
                word_edit_distance = nltk.edit_distance(ref_words, hyp_words)
                word_errors += word_edit_distance
        
        wer = word_errors / total_words if total_words > 0 else 0
        summary['wer'] = wer
        
        # Extract references and hypotheses for batch processing
        references = []
        hypotheses = []
        
        for result in all_results:
            if 'reference' in result and 'hypothesis' in result:
                references.append(result['reference'])
                hypotheses.append(result['hypothesis'])
        
        # Load model for semantic similarity metrics (if needed)
        try:
            semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Calculate Semantic Similarity
            ref_embeddings = semantic_model.encode(references)
            hyp_embeddings = semantic_model.encode(hypotheses)
            
            # Compute cosine similarity between reference and hypothesis embeddings
            semantic_similarities = []
            for i in range(len(references)):
                ref_emb = ref_embeddings[i]
                hyp_emb = hyp_embeddings[i]
                
                # Compute cosine similarity
                similarity = np.dot(ref_emb, hyp_emb) / (np.linalg.norm(ref_emb) * np.linalg.norm(hyp_emb))
                semantic_similarities.append(similarity)
            
            avg_semantic_similarity = np.mean(semantic_similarities)
            summary['semantic_similarity'] = float(avg_semantic_similarity)
            
            # Calculate Semantic WER (percentage of incorrect words based on semantic threshold)
            semantic_wer_threshold = 0.7  # Threshold for semantic similarity
            semantic_errors = 0
            
            for ref, hyp in zip(references, hypotheses):
                ref_words = ref.split()
                hyp_words = hyp.split()
                
                # Skip empty sequences
                if len(ref_words) == 0:
                    continue
                
                # Calculate word-by-word semantic similarity
                word_errors = 0
                for ref_word in ref_words:
                    # Check if any word in hypothesis is semantically similar to reference word
                    word_found = False
                    for hyp_word in hyp_words:
                        # Calculate semantic similarity between words
                        try:
                            ref_word_emb = semantic_model.encode([ref_word])[0]
                            hyp_word_emb = semantic_model.encode([hyp_word])[0]
                            word_sim = np.dot(ref_word_emb, hyp_word_emb) / (np.linalg.norm(ref_word_emb) * np.linalg.norm(hyp_word_emb))
                            
                            if word_sim > semantic_wer_threshold:
                                word_found = True
                                break
                        except:
                            pass
                    
                    if not word_found:
                        word_errors += 1
                
                semantic_errors += word_errors
            
            semantic_wer = semantic_errors / total_words if total_words > 0 else 0
            summary['semantic_wer'] = float(semantic_wer)
            
        except Exception as e:
            print(f"Warning: Could not calculate semantic metrics: {str(e)}")
            summary['semantic_similarity'] = 0.0
            summary['semantic_wer'] = 0.0
        
        # Calculate BERTScore
        try:
            # Calculate BERTScore
            P, R, F1 = bert_score.score(hypotheses, references, lang="en", verbose=False)
            
            summary['bertscore_precision'] = float(torch.mean(P).item())
            summary['bertscore_recall'] = float(torch.mean(R).item())
            summary['bertscore_f1'] = float(torch.mean(F1).item())
        except Exception as e:
            print(f"Warning: Could not calculate BERTScore: {str(e)}")
            summary['bertscore_precision'] = 0.0
            summary['bertscore_recall'] = 0.0
            summary['bertscore_f1'] = 0.0
        
        # Calculate word similarity (using SequenceMatcher)
        word_similarities = []
        for ref, hyp in zip(references, hypotheses):
            similarity = SequenceMatcher(None, ref, hyp).ratio()
            word_similarities.append(similarity)
        
        summary['word_similarity'] = float(np.mean(word_similarities))
        
        # Calculate METEOR score
        try:
            meteor_scores = []
            for ref, hyp in zip(references, hypotheses):
                ref_tokens = [ref.split()]  # METEOR expects a list of reference tokenized sentences
                hyp_tokens = hyp.split()
                
                score = meteor_score.meteor_score(ref_tokens, hyp_tokens)
                meteor_scores.append(score)
            
            summary['meteor_score'] = float(np.mean(meteor_scores))
        except Exception as e:
            print(f"Warning: Could not calculate METEOR score: {str(e)}")
            summary['meteor_score'] = 0.0
        
        # Calculate ROUGE scores
        try:
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            
            rouge1_scores = []
            rouge2_scores = []
            rougeL_scores = []
            
            for ref, hyp in zip(references, hypotheses):
                scores = scorer.score(ref, hyp)
                
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
            
            summary['rouge1_score'] = float(np.mean(rouge1_scores))
            summary['rouge2_score'] = float(np.mean(rouge2_scores))
            summary['rougeL_score'] = float(np.mean(rougeL_scores))
        except Exception as e:
            print(f"Warning: Could not calculate ROUGE scores: {str(e)}")
            summary['rouge1_score'] = 0.0
            summary['rouge2_score'] = 0.0
            summary['rougeL_score'] = 0.0
        
        # Calculate BLEU scores
        try:
            # Get smoothing function for BLEU
            smoothie = SmoothingFunction().method1
            
            # Sentence BLEU - Average of sentence-level BLEU scores
            sentence_bleu_scores = []
            for ref, hyp in zip(references, hypotheses):
                ref_tokens = [ref.split()]  # BLEU expects a list of reference tokenized sentences
                hyp_tokens = hyp.split()
                
                # Skip empty sequences
                if len(hyp_tokens) == 0 or all(len(r) == 0 for r in ref_tokens):
                    continue
                
                # Use smoothing function to handle cases with no n-gram overlaps
                score = sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=smoothie)
                sentence_bleu_scores.append(score)
            
            summary['sentence_bleu_score'] = float(np.mean(sentence_bleu_scores) * 100)  # Convert to percentage
            
            # Corpus BLEU - BLEU score over the entire corpus
            tokenized_refs = [[r.split()] for r in references]  # List of lists of tokenized references
            tokenized_hyps = [h.split() for h in hypotheses]  # List of tokenized hypotheses
            
            corpus_bleu_score = corpus_bleu(tokenized_refs, tokenized_hyps, smoothing_function=smoothie)
            summary['corpus_bleu_score'] = float(corpus_bleu_score * 100)  # Convert to percentage
        except Exception as e:
            print(f"Warning: Could not calculate BLEU scores: {str(e)}")
            summary['sentence_bleu_score'] = 0.0
            summary['corpus_bleu_score'] = 0.0
        
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
            f.write(f"- Average Phonetic Edit Distance: {summary['avg_phonetic_distance']:.3f} ± {summary['phonetic_distance_std']:.3f} (lower is better)\n")
            f.write(f"- Character Error Rate (CER): {summary['cer']:.3f} (lower is better)\n")
            f.write(f"- Word Error Rate (WER): {summary['wer']:.3f} (lower is better)\n")
            f.write(f"- Semantic WER: {summary['semantic_wer']:.3f} (lower is better)\n")
            f.write(f"- Semantic Similarity: {summary['semantic_similarity']:.3f} (higher is better)\n\n")
            
            f.write("Text Similarity Metrics:\n")
            f.write(f"- BERTScore Precision: {summary['bertscore_precision']:.3f}\n")
            f.write(f"- BERTScore Recall: {summary['bertscore_recall']:.3f}\n")
            f.write(f"- BERTScore F1: {summary['bertscore_f1']:.3f}\n")
            f.write(f"- Word Similarity: {summary['word_similarity']:.3f}\n")
            f.write(f"- METEOR Score: {summary['meteor_score']:.3f}\n")
            f.write(f"- ROUGE-1 Score: {summary['rouge1_score']:.3f}\n")
            f.write(f"- ROUGE-2 Score: {summary['rouge2_score']:.3f}\n")
            f.write(f"- ROUGE-L Score: {summary['rougeL_score']:.3f}\n")
            f.write(f"- Sentence BLEU Score: {summary['sentence_bleu_score']:.2f}\n")
            f.write(f"- Corpus BLEU Score: {summary['corpus_bleu_score']:.2f}\n\n")
            
            f.write("Operation Counts:\n")
            for op, count in op_counts.items():
                f.write(f"- {op.capitalize()}: {count} ({summary['operations_percent'][op]:.1f}%)\n")
        
        return summary

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

    def get_closest_viseme(self, phoneme):
        """
        Map any phoneme to the closest viseme in Microsoft's 22-category system,
        even if it's not explicitly defined in the mapping.
        
        Args:
            phoneme: The phoneme to map
            
        Returns:
            int: The viseme ID (0-21) that best matches this phoneme
            
        Raises:
            ValueError: If the phoneme cannot be mapped to any viseme category
        """
        # First check if the phoneme is already in our mapping
        if phoneme in self.phoneme_to_viseme:
            return self.phoneme_to_viseme[phoneme]
        
        # Handle phonemes with stress markers (e.g., "ow2", "ae1")
        # First try stripping any trailing digits
        base_phoneme = phoneme.rstrip('0123456789')
        if base_phoneme and base_phoneme in self.phoneme_to_viseme:
            return self.phoneme_to_viseme[base_phoneme]
            
        # If it's a special character, map to silence
        if not phoneme.isalpha():
            return 0  # silence
            
        # For consonants, map based on place of articulation
        consonant_features = {
            # Bilabial: p, b, m
            'bilabial': ['p', 'b', 'm'],
            # Labiodental: f, v
            'labiodental': ['f', 'v'],
            # Dental: θ, ð
            'dental': ['θ', 'ð', 'th', 'dh'],
            # Alveolar: t, d, s, z, n, l
            'alveolar_stop': ['t', 'd', 'n'],
            'alveolar_fricative': ['s', 'z'],
            'alveolar_liquid': ['l'],
            # Post-alveolar: ʃ, ʒ, tʃ, dʒ
            'postalveolar': ['ʃ', 'ʒ', 'tʃ', 'dʒ', 'sh', 'zh', 'ch', 'j'],
            # Retroflex/rhotic: r, ɹ, ɾ
            'retroflex': ['r', 'ɹ', 'ɾ'],
            # Palatal: j
            'palatal': ['j', 'y'],
            # Velar: k, g, ŋ, w
            'velar': ['k', 'g', 'ŋ', 'ng', 'w'],
            # Glottal: h, ʔ
            'glottal': ['h', 'ʔ']
        }
        
        # Vowel features
        vowel_features = {
            # Front vowels: i, ɪ, e, ɛ, æ
            'front_close': ['i', 'ɪ', 'y'],
            'front_mid': ['e', 'ɛ', 'ə'],
            'front_open': ['æ', 'a'],
            # Central vowels: ə, ʌ, ɜ
            'central': ['ə', 'ʌ', 'ɜ', 'ɐ'],
            # R-colored: ɝ, ɚ
            'r_colored': ['ɝ', 'ɚ', 'ɻ'],
            # Back vowels: u, ʊ, o, ɔ, ɑ, ɒ
            'back_close': ['u', 'ʊ'],
            'back_mid': ['o', 'ɔ'],
            'back_open': ['ɑ', 'ɒ']
        }
        
        # Diphthongs
        diphthongs = {
            'diphthong_a': ['aɪ', 'aj'],
            'diphthong_o': ['ɔɪ', 'oj'],
            'diphthong_au': ['aʊ', 'aw']
        }
        
        # Clean the phoneme for comparison
        clean_phoneme = phoneme.lower().strip()
        
        # Check consonants
        for group, phonemes in consonant_features.items():
            if any(clean_phoneme == p or clean_phoneme.startswith(p) for p in phonemes):
                if group == 'bilabial':
                    return 21  # p, b, m
                elif group == 'labiodental':
                    return 18  # f, v
                elif group == 'dental':
                    if clean_phoneme == 'ð' or clean_phoneme == 'dh':
                        return 17  # ð
                    else:
                        return 19  # θ with alveolar stops
                elif group == 'alveolar_stop' or group == 'alveolar_fricative':
                    if clean_phoneme in ['s', 'z']:
                        return 15  # s, z
                    else:
                        return 19  # t, d, n, θ
                elif group == 'alveolar_liquid':
                    return 14  # l
                elif group == 'postalveolar':
                    return 16  # ʃ, ʒ, tʃ, dʒ
                elif group == 'retroflex':
                    return 13  # ɹ
                elif group == 'palatal':
                    return 6   # j with front vowels
                elif group == 'velar':
                    if clean_phoneme == 'w':
                        return 7   # w with /u/
                    else:
                        return 20  # k, g, ŋ
                elif group == 'glottal':
                    return 12  # h
        
        # Check vowels
        for group, phonemes in vowel_features.items():
            if any(clean_phoneme == p or clean_phoneme.startswith(p) for p in phonemes):
                if group == 'front_close' or group == 'palatal':
                    return 6   # i, ɪ, j
                elif group == 'front_mid':
                    return 4   # ɛ, e
                elif group == 'front_open':
                    return 1   # æ, a
                elif group == 'central':
                    return 1   # ə, ʌ
                elif group == 'r_colored':
                    return 5   # ɝ
                elif group == 'back_close':
                    return 7   # u, ʊ
                elif group == 'back_mid':
                    return 8   # o, ɔ
                elif group == 'back_open':
                    return 2   # ɑ
        
        # Check diphthongs
        for group, phonemes in diphthongs.items():
            if any(clean_phoneme == p or clean_phoneme.startswith(p) for p in phonemes):
                if group == 'diphthong_a':
                    return 11  # aɪ
                elif group == 'diphthong_o':
                    return 10  # ɔɪ
                elif group == 'diphthong_au':
                    return 9   # aʊ
        
        # Raise an error if we can't categorize the phoneme
        raise ValueError(f"Could not map phoneme '{phoneme}' to any viseme category")

# Add this class after the LipReadingEvaluator class, before the main() function

class ModelComparator:
    """
    Class for comparing the performance of two different models on visual speech recognition tasks.
    Takes in two JSON files containing evaluation results and creates comparison visualizations.
    """
    
    def __init__(self, json_file1, json_file2, model1_name=None, model2_name=None):
        """
        Initialize the ModelComparator with two JSON result files
        
        Parameters:
        - json_file1: Path to the first model's JSON results file
        - json_file2: Path to the second model's JSON results file
        - model1_name: Name to use for the first model (default: derived from filename)
        - model2_name: Name to use for the second model (default: derived from filename)
        """
        self.json_file1 = json_file1
        self.json_file2 = json_file2
        
        # Set model names based on filenames if not provided
        if model1_name is None:
            self.model1_name = os.path.basename(os.path.dirname(json_file1))
        else:
            self.model1_name = model1_name
            
        if model2_name is None:
            self.model2_name = os.path.basename(os.path.dirname(json_file2))
        else:
            self.model2_name = model2_name
        
        # Load data from JSON files
        self.load_data()
    
    def load_data(self):
        """Load data from the JSON files"""
        try:
            # Load summary data
            with open(self.json_file1, 'r') as f:
                data1 = json.load(f)
                
            with open(self.json_file2, 'r') as f:
                data2 = json.load(f)
            
            # Initialize summary dictionaries
            self.summary1 = {}
            self.summary2 = {}
            self.examples1 = []
            self.examples2 = []
            
            # Check if this is a hypo-xxxxx.json file (contains 'ref' and 'hypo' arrays)
            # or a summary.json file (contains metrics directly)
            if 'ref' in data1 and 'hypo' in data1:
                # This is a hypothesis file with references and hypotheses
                print(f"Detected hypothesis file format. Processing examples...")
                
                # Extract references and hypotheses
                refs1 = data1.get('ref', [])
                hyps1 = data1.get('hypo', [])
                utt_ids1 = data1.get('utt_id', [])
                
                refs2 = data2.get('ref', [])
                hyps2 = data2.get('hypo', [])
                utt_ids2 = data2.get('utt_id', [])
                
                # Make sure we have matching references and hypotheses
                if len(refs1) != len(hyps1) or len(refs2) != len(hyps2):
                    print(f"Warning: Mismatch in number of references and hypotheses")
                
                # Create examples for each model
                for i in range(min(len(refs1), len(hyps1))):
                    utt_id = utt_ids1[i] if i < len(utt_ids1) else f"example_{i}"
                    self.examples1.append({
                        'reference': refs1[i],
                        'hypothesis': hyps1[i],
                        'utt_id': utt_id
                    })
                
                for i in range(min(len(refs2), len(hyps2))):
                    utt_id = utt_ids2[i] if i < len(utt_ids2) else f"example_{i}"
                    self.examples2.append({
                        'reference': refs2[i],
                        'hypothesis': hyps2[i],
                        'utt_id': utt_id
                    })
                
                # Calculate metrics for both models
                self._calculate_metrics_from_examples()
                
            else:
                # This is a summary.json file
                self.summary1 = data1
                self.summary2 = data2
                
                # Check if we have examples data
                if 'examples' in self.summary1:
                    self.examples1 = self.summary1['examples']
                
                if 'examples' in self.summary2:
                    self.examples2 = self.summary2['examples']
            
            print(f"Loaded data from {self.json_file1} and {self.json_file2}")
            print(f"Model 1: {len(self.examples1)} examples, Model 2: {len(self.examples2)} examples")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            self.summary1 = {}
            self.summary2 = {}
            self.examples1 = []
            self.examples2 = []
    
    def _calculate_metrics_from_examples(self):
        """Calculate metrics from examples when using hypothesis files"""
        try:
            # Calculate metrics for model 1
            self._calculate_model_metrics(self.examples1, self.summary1)
            
            # Calculate metrics for model 2
            self._calculate_model_metrics(self.examples2, self.summary2)
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
    
    def _calculate_model_metrics(self, examples, summary):
        """Calculate metrics for a single model from its examples"""
        if not examples:
            return
        
        # Extract references and hypotheses
        references = [ex['reference'] for ex in examples]
        hypotheses = [ex['hypothesis'] for ex in examples]
        
        # Calculate Character Error Rate (CER)
        total_chars = sum(len(ref) for ref in references)
        char_errors = sum(nltk.edit_distance(ref, hyp) for ref, hyp in zip(references, hypotheses))
        cer = char_errors / total_chars if total_chars > 0 else 0
        summary['cer'] = cer
        
        # Calculate Word Error Rate (WER)
        ref_words = [ref.split() for ref in references]
        hyp_words = [hyp.split() for hyp in hypotheses]
        total_words = sum(len(words) for words in ref_words)
        word_errors = sum(nltk.edit_distance(ref, hyp) for ref, hyp in zip(ref_words, hyp_words))
        wer = word_errors / total_words if total_words > 0 else 0
        summary['wer'] = wer
        
        # Calculate word similarity (using SequenceMatcher)
        word_similarities = [SequenceMatcher(None, ref, hyp).ratio() for ref, hyp in zip(references, hypotheses)]
        summary['word_similarity'] = float(np.mean(word_similarities))
        
        try:
            # Calculate METEOR score
            meteor_scores = []
            for ref, hyp in zip(references, hypotheses):
                ref_tokens = [ref.split()]  # METEOR expects a list of reference tokenized sentences
                hyp_tokens = hyp.split()
                score = meteor_score.meteor_score(ref_tokens, hyp_tokens)
                meteor_scores.append(score)
            summary['meteor_score'] = float(np.mean(meteor_scores))
        except Exception as e:
            print(f"Warning: Could not calculate METEOR score: {e}")
            summary['meteor_score'] = 0.0
        
        try:
            # Calculate ROUGE scores
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            rouge1_scores = []
            rouge2_scores = []
            rougeL_scores = []
            for ref, hyp in zip(references, hypotheses):
                scores = scorer.score(ref, hyp)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
            summary['rouge1_score'] = float(np.mean(rouge1_scores))
            summary['rouge2_score'] = float(np.mean(rouge2_scores))
            summary['rougeL_score'] = float(np.mean(rougeL_scores))
        except Exception as e:
            print(f"Warning: Could not calculate ROUGE scores: {e}")
            summary['rouge1_score'] = 0.0
            summary['rouge2_score'] = 0.0
            summary['rougeL_score'] = 0.0
        
        try:
            # Calculate BLEU scores
            smoothie = SmoothingFunction().method1
            # Sentence BLEU
            sentence_bleu_scores = []
            for ref, hyp in zip(references, hypotheses):
                ref_tokens = [ref.split()]
                hyp_tokens = hyp.split()
                if not hyp_tokens or not ref_tokens[0]:
                    continue
                score = sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=smoothie)
                sentence_bleu_scores.append(score)
            summary['sentence_bleu_score'] = float(np.mean(sentence_bleu_scores) * 100)
            
            # Corpus BLEU
            tokenized_refs = [[r.split()] for r in references]
            tokenized_hyps = [h.split() for h in hypotheses]
            corpus_bleu_score = corpus_bleu(tokenized_refs, tokenized_hyps, smoothing_function=smoothie)
            summary['corpus_bleu_score'] = float(corpus_bleu_score * 100)
        except Exception as e:
            print(f"Warning: Could not calculate BLEU scores: {e}")
            summary['sentence_bleu_score'] = 0.0
            summary['corpus_bleu_score'] = 0.0
        
        try:
            # Calculate BERTScore
            P, R, F1 = bert_score.score(hypotheses, references, lang="en", verbose=False)
            summary['bertscore_precision'] = float(torch.mean(P).item())
            summary['bertscore_recall'] = float(torch.mean(R).item())
            summary['bertscore_f1'] = float(torch.mean(F1).item())
        except Exception as e:
            print(f"Warning: Could not calculate BERTScore: {e}")
            summary['bertscore_precision'] = 0.0
            summary['bertscore_recall'] = 0.0
            summary['bertscore_f1'] = 0.0
        
        try:
            # Calculate Semantic similarity
            semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            ref_embeddings = semantic_model.encode(references)
            hyp_embeddings = semantic_model.encode(hypotheses)
            
            # Compute cosine similarity
            semantic_similarities = []
            for i in range(len(references)):
                ref_emb = ref_embeddings[i]
                hyp_emb = hyp_embeddings[i]
                similarity = np.dot(ref_emb, hyp_emb) / (np.linalg.norm(ref_emb) * np.linalg.norm(hyp_emb))
                semantic_similarities.append(similarity)
            
            summary['semantic_similarity'] = float(np.mean(semantic_similarities))
            
            # Calculate Semantic WER (simple approximation)
            semantic_wer = 1.0 - summary['semantic_similarity']
            summary['semantic_wer'] = float(semantic_wer)
        except Exception as e:
            print(f"Warning: Could not calculate semantic metrics: {e}")
            summary['semantic_similarity'] = 0.0
            summary['semantic_wer'] = 0.0
        
        # Add viseme alignment score and phonetic edit distance calculation
        try:
            print("Calculating viseme alignment and phonetic edit distance metrics...")
            evaluator = LipReadingEvaluator()
            
            viseme_scores = []
            phonetic_distances = []
            
            # Process each example pair
            for ref, hyp in zip(references, hypotheses):
                if not ref or not hyp:
                    continue
                    
                # Calculate metrics using LipReadingEvaluator
                results = evaluator.evaluate_pair(ref, hyp)
                
                # Extract scores
                if 'viseme_alignment_score' in results:
                    viseme_scores.append(results['viseme_alignment_score'])
                
                if 'phonetic_edit_distance' in results:
                    phonetic_distances.append(results['phonetic_edit_distance'])
            
            # Calculate average scores if we have enough data
            if viseme_scores:
                summary['viseme_alignment_score'] = float(np.mean(viseme_scores))
            else:
                summary['viseme_alignment_score'] = 0.0
                
            if phonetic_distances:
                summary['phonetic_edit_distance'] = float(np.mean(phonetic_distances))
            else:
                summary['phonetic_edit_distance'] = 0.0
                
        except Exception as e:
            print(f"Warning: Could not calculate viseme and phonetic metrics: {e}")
            import traceback
            traceback.print_exc()
            summary['viseme_alignment_score'] = 0.0
            summary['phonetic_edit_distance'] = 0.0
    
    def create_comparison_plots(self, output_dir):
        """
        Create all comparison plots and save them to the output directory
        
        Parameters:
        - output_dir: Directory to save visualizations
        
        Returns:
        - Dictionary with paths to created visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nCreating model comparison visualizations in: {output_dir}")
        
        # Dictionary to store visualization paths
        viz_files = {}
        
        # 1. Error Rate Comparison Bar Chart
        viz_files['error_rates'] = self.plot_error_rate_comparison(output_dir)
        
        # 2. Per-Example Performance Scatter (if we have example data)
        if self.examples1 and self.examples2:
            viz_files['performance_scatter'] = self.plot_performance_scatter(output_dir)
        
        # 3. Confusion Matrix Difference (if available in data)
        if all(key in self.summary1 for key in ['confusion_matrix', 'viseme_classes']) and \
           all(key in self.summary2 for key in ['confusion_matrix', 'viseme_classes']):
            viz_files['confusion_diff'] = self.plot_confusion_matrix_difference(output_dir)
        
        # 4. Word Error Analysis
        viz_files['word_error'] = self.plot_word_error_analysis(output_dir)
        
        # 5. Length-Based Performance Curves (if we have example data)
        if self.examples1 and self.examples2:
            viz_files['length_curves'] = self.plot_length_performance_curves(output_dir)
        
        # Create comparison summary
        viz_files['summary'] = self.generate_comparison_summary(output_dir)
        
        return viz_files
    
    def plot_error_rate_comparison(self, output_dir):
        """
        Create bar chart comparing error rates between models
        
        Parameters:
        - output_dir: Directory to save visualization
        
        Returns:
        - Path to saved visualization
        """
        try:
            # Define metrics to compare (lower is better)
            lower_better_metrics = [
                ('wer', 'Word Error Rate'),
                ('cer', 'Character Error Rate'),
                ('semantic_wer', 'Semantic WER'),
                ('phonetic_edit_distance', 'Phonetic Edit Distance')
            ]
            
            # Define metrics to compare (higher is better)
            higher_better_metrics = [
                ('bertscore_f1', 'BERTScore F1'),
                ('semantic_similarity', 'Semantic Similarity'),
                ('word_similarity', 'Word Similarity'),
                ('viseme_alignment_score', 'Viseme Alignment Score'),
                ('meteor_score', 'METEOR Score'),
                ('rouge1_score', 'ROUGE-1'),
                ('sentence_bleu_score', 'BLEU Score')
            ]
            
            # Create figure with subplots for each metric type
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
            
            # Plot metrics where lower is better
            self._plot_metrics_group(ax1, lower_better_metrics, 'Error Rates (Lower is Better)')
            
            # Plot metrics where higher is better
            self._plot_metrics_group(ax2, higher_better_metrics, 'Similarity Scores (Higher is Better)')
            
            # Add overall title
            fig.suptitle(f'Model Comparison: {self.model1_name} vs. {self.model2_name}', fontsize=16)
            
            plt.tight_layout()
            
            # Save figure
            output_file = os.path.join(output_dir, 'error_rate_comparison.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_file
            
        except Exception as e:
            print(f"Error creating error rate comparison: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _plot_metrics_group(self, ax, metrics, title):
        """Helper method to plot a group of metrics on an axis"""
        # Get values for each metric
        labels = []
        model1_values = []
        model2_values = []
        
        for metric_key, metric_name in metrics:
            # Only include metrics that exist in both summaries
            if metric_key in self.summary1 and metric_key in self.summary2:
                # Extract scalar values only
                val1 = self.summary1[metric_key]
                val2 = self.summary2[metric_key]
                
                # Skip if not a scalar value
                if isinstance(val1, (list, dict, np.ndarray)) or isinstance(val2, (list, dict, np.ndarray)):
                    continue
                    
                labels.append(metric_name)
                model1_values.append(val1)
                model2_values.append(val2)
        
        if not labels:
            ax.text(0.5, 0.5, "No metrics available", ha='center', va='center')
            ax.set_title(title)
            return
        
        # Convert to numpy arrays to ensure consistent types
        model1_values = np.array(model1_values)
        model2_values = np.array(model2_values)
        
        # Set width of bars
        bar_width = 0.35
        x = np.arange(len(labels))
        
        # Create bars
        ax.bar(x - bar_width/2, model1_values, bar_width, label=self.model1_name, color='royalblue')
        ax.bar(x + bar_width/2, model2_values, bar_width, label=self.model2_name, color='darkorange')
        
        # Add labels and legend
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        
        # Add value labels on bars
        for i, v in enumerate(model1_values):
            ax.text(i - bar_width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        
        for i, v in enumerate(model2_values):
            ax.text(i + bar_width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    
    def plot_performance_scatter(self, output_dir):
        """
        Create a scatter plot showing performance on individual examples
        
        Parameters:
        - output_dir: Directory to save visualization
        
        Returns:
        - Path to saved visualization
        """
        try:
            # Need examples for both models
            if not self.examples1 or not self.examples2:
                print("Warning: Not enough example data for performance scatter plot")
                return None
                
            # Check for minimum number of examples
            min_examples = 5
            if len(self.examples1) < min_examples or len(self.examples2) < min_examples:
                print(f"Warning: Need at least {min_examples} examples for meaningful performance scatter")
                print(f"Found: Model 1: {len(self.examples1)}, Model 2: {len(self.examples2)}")
                return None
            
            # Calculate scores for each example if not already present
            # Use word similarity as our metric since it's less expensive than semantic metrics
            scores1 = []
            scores2 = []
            references = []
            
            # We need matching references, so build a common set
            common_refs = {}
            
            # First build dictionary of utt_id -> reference and scores
            for ex in self.examples1:
                if 'utt_id' in ex and 'reference' in ex and 'hypothesis' in ex:
                    utt_id = ex['utt_id']
                    ref = ex['reference']
                    hyp = ex['hypothesis']
                    
                    # Calculate similarity score if not already present
                    if 'word_similarity' not in ex:
                        score = SequenceMatcher(None, ref, hyp).ratio()
                    else:
                        score = ex['word_similarity']
                        
                    common_refs[utt_id] = {
                        'reference': ref,
                        'model1_score': score,
                        'model1_hyp': hyp
                    }
            
            # Now match with model 2 and keep only common references
            for ex in self.examples2:
                if 'utt_id' in ex and 'reference' in ex and 'hypothesis' in ex:
                    utt_id = ex['utt_id']
                    if utt_id in common_refs:
                        ref = ex['reference']
                        hyp = ex['hypothesis']
                        
                        # Calculate similarity score if not already present
                        if 'word_similarity' not in ex:
                            score = SequenceMatcher(None, ref, hyp).ratio()
                        else:
                            score = ex['word_similarity']
                            
                        common_refs[utt_id]['model2_score'] = score
                        common_refs[utt_id]['model2_hyp'] = hyp
            
            # Keep only entries that have both model scores
            valid_examples = [
                item for item in common_refs.values() 
                if 'model1_score' in item and 'model2_score' in item
            ]
            
            # Extract scores
            if not valid_examples:
                print("Warning: No matching examples between models. Cannot create performance scatter.")
                return None
                
            scores1 = [item['model1_score'] for item in valid_examples]
            scores2 = [item['model2_score'] for item in valid_examples]
            references = [item['reference'] for item in valid_examples]
            
            # Create scatter plot
            plt.figure(figsize=(10, 10))
            plt.scatter(scores1, scores2, alpha=0.6)
            
            # Add diagonal line
            max_score = max(max(scores1), max(scores2)) if scores1 and scores2 else 1.0
            plt.plot([0, max_score], [0, max_score], 'r--')
            
            # Add labels and title
            plt.xlabel(f'{self.model1_name} Score')
            plt.ylabel(f'{self.model2_name} Score')
            plt.title(f'Per-Example Performance Comparison')
            
            # Count points above/below diagonal
            better1 = sum(1 for a, b in zip(scores1, scores2) if a > b)
            better2 = sum(1 for a, b in zip(scores1, scores2) if b > a)
            equal = sum(1 for a, b in zip(scores1, scores2) if a == b)
            
            # Add annotated regions
            plt.annotate(f'{self.model1_name} better\n({better1} examples)', 
                       xy=(0.75, 0.25), xycoords='axes fraction',
                       ha='center', va='center', fontsize=12)
            
            plt.annotate(f'{self.model2_name} better\n({better2} examples)', 
                       xy=(0.25, 0.75), xycoords='axes fraction',
                       ha='center', va='center', fontsize=12)
            
            # Add stats in the corner
            avg_diff = np.mean([a - b for a, b in zip(scores1, scores2)])
            plt.figtext(0.02, 0.02, 
                      f'Average score difference: {avg_diff:.4f}\n' +
                      f'Equal performance: {equal} examples',
                      ha='left', fontsize=10)
            
            # Save the figure
            output_file = os.path.join(output_dir, 'per_example_performance.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_file
            
        except Exception as e:
            print(f"Error creating performance scatter plot: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def plot_confusion_matrix_difference(self, output_dir):
        """
        Create a heatmap showing the difference between confusion matrices
        
        Parameters:
        - output_dir: Directory to save visualization
        
        Returns:
        - Path to saved visualization
        """
        try:
            # Get confusion matrices
            cm1 = np.array(self.summary1['confusion_matrix'])
            cm2 = np.array(self.summary2['confusion_matrix'])
            
            # Get viseme classes
            classes1 = self.summary1['viseme_classes']
            classes2 = self.summary2['viseme_classes']
            
            # Ensure dimensions match
            if cm1.shape != cm2.shape or classes1 != classes2:
                print("Warning: Confusion matrices or classes don't match between models")
                return None
            
            # Calculate the difference (Model1 - Model2)
            diff_matrix = cm1 - cm2
            
            # Create visualization
            plt.figure(figsize=(12, 10))
            
            # Use a diverging colormap centered at 0
            cmap = plt.cm.RdBu_r
            
            # Determine the maximum absolute difference for symmetrical color scaling
            vmax = max(abs(np.min(diff_matrix)), abs(np.max(diff_matrix)))
            vmin = -vmax
            
            # Create heatmap
            im = plt.imshow(diff_matrix, cmap=cmap, vmin=vmin, vmax=vmax)
            
            # Add colorbar
            cbar = plt.colorbar(im)
            cbar.set_label(f'{self.model1_name} - {self.model2_name}')
            
            # Add labels and ticks
            plt.xlabel('Predicted Viseme')
            plt.ylabel('True Viseme')
            plt.title('Confusion Matrix Difference')
            
            # Set tick labels
            plt.xticks(range(len(classes1)), classes1, rotation=90)
            plt.yticks(range(len(classes1)), classes1)
            
            # Add text annotations
            for i in range(len(classes1)):
                for j in range(len(classes1)):
                    value = diff_matrix[i, j]
                    if abs(value) > vmax/10:  # Only show significant differences
                        text_color = 'white' if abs(value) > vmax/2 else 'black'
                        plt.text(j, i, f'{value:.1f}', ha='center', va='center', 
                                color=text_color, fontsize=8)
            
            plt.tight_layout()
            
            # Save figure
            output_file = os.path.join(output_dir, 'confusion_matrix_difference.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_file
            
        except Exception as e:
            print(f"Error creating confusion matrix difference: {e}")
            return None
    
    def plot_word_error_analysis(self, output_dir):
        """
        Create visualization showing error analysis by word
        
        Parameters:
        - output_dir: Directory to save visualization
        
        Returns:
        - Path to saved visualization
        """
        try:
            # We need at least some examples for both models
            if not self.examples1 or not self.examples2:
                print("Warning: Not enough example data for word error analysis")
                return None
            
            # Set a minimum threshold for useful analysis
            min_examples = 5
            if len(self.examples1) < min_examples or len(self.examples2) < min_examples:
                print(f"Warning: Need at least {min_examples} examples for meaningful word error analysis")
                print(f"Found: Model 1: {len(self.examples1)}, Model 2: {len(self.examples2)}")
                return None
                
            # Extract references and hypotheses
            refs1 = [ex.get('reference', '') for ex in self.examples1]
            hyps1 = [ex.get('hypothesis', '') for ex in self.examples1]
            
            refs2 = [ex.get('reference', '') for ex in self.examples2]
            hyps2 = [ex.get('hypothesis', '') for ex in self.examples2]
            
            # Get missing words for each model
            missing_words1 = Counter()
            missing_words2 = Counter()
            
            for ref, hyp in zip(refs1, hyps1):
                if not ref or not hyp:  # Skip empty examples
                    continue
                ref_words = set(ref.lower().split())
                hyp_words = set(hyp.lower().split())
                for word in ref_words - hyp_words:
                    missing_words1[word] += 1
                    
            for ref, hyp in zip(refs2, hyps2):
                if not ref or not hyp:  # Skip empty examples
                    continue
                ref_words = set(ref.lower().split())
                hyp_words = set(hyp.lower().split())
                for word in ref_words - hyp_words:
                    missing_words2[word] += 1
            
            # If we don't have enough missing words, can't do analysis
            if len(missing_words1) < 3 or len(missing_words2) < 3:
                print("Warning: Not enough missed words for word error analysis")
                return None
            
            # Get top N words from each
            top_n = min(15, max(len(missing_words1), len(missing_words2)))
            common_words = set([w for w, _ in missing_words1.most_common(top_n)] + 
                              [w for w, _ in missing_words2.most_common(top_n)])
            
            # Create a comparison of these common words
            words = list(common_words)
            model1_counts = [missing_words1.get(w, 0) for w in words]
            model2_counts = [missing_words2.get(w, 0) for w in words]
            
            # Create a horizontal bar chart
            plt.figure(figsize=(12, max(8, len(words) * 0.3)))
            
            # Sort by total error count
            total_counts = [a + b for a, b in zip(model1_counts, model2_counts)]
            sorted_indices = np.argsort(total_counts)[::-1]  # Descending order
            
            words = [words[i] for i in sorted_indices]
            model1_counts = [model1_counts[i] for i in sorted_indices]
            model2_counts = [model2_counts[i] for i in sorted_indices]
            
            # Limit to top 15 for readability
            if len(words) > 15:
                words = words[:15]
                model1_counts = model1_counts[:15]
                model2_counts = model2_counts[:15]
            
            # Plot horizontal bars
            y_pos = np.arange(len(words))
            
            plt.barh(y_pos - 0.2, model1_counts, 0.4, label=self.model1_name, color='royalblue')
            plt.barh(y_pos + 0.2, model2_counts, 0.4, label=self.model2_name, color='darkorange')
            
            plt.yticks(y_pos, words)
            plt.xlabel('Number of Errors')
            plt.title('Most Frequently Missed Words by Model')
            plt.legend()
            
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            # Save figure
            output_file = os.path.join(output_dir, 'word_error_analysis.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_file
            
        except Exception as e:
            print(f"Error creating word error analysis: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def plot_length_performance_curves(self, output_dir):
        """
        Create visualization showing performance by utterance length
        
        Parameters:
        - output_dir: Directory to save visualization
        
        Returns:
        - Path to saved visualization
        """
        try:
            # Need examples for both models
            if not self.examples1 or not self.examples2:
                print("Warning: Not enough example data for length performance curves")
                return None
                
            # Check for minimum number of examples
            min_examples = 10
            if len(self.examples1) < min_examples or len(self.examples2) < min_examples:
                print(f"Warning: Need at least {min_examples} examples for meaningful length curves")
                print(f"Found: Model 1: {len(self.examples1)}, Model 2: {len(self.examples2)}")
                return None
            
            # Group examples by length
            length_scores1 = defaultdict(list)
            length_scores2 = defaultdict(list)
            
            # Process examples from model 1
            for ex in self.examples1:
                if 'reference' in ex and 'hypothesis' in ex:
                    # Use word similarity as our metric if not already present
                    if 'word_similarity' not in ex:
                        score = SequenceMatcher(None, ex['reference'], ex['hypothesis']).ratio()
                    else:
                        score = ex['word_similarity']
                        
                    length = len(ex['reference'].split())
                    length_scores1[length].append(score)
            
            # Process examples from model 2
            for ex in self.examples2:
                if 'reference' in ex and 'hypothesis' in ex:
                    # Use word similarity as our metric if not already present
                    if 'word_similarity' not in ex:
                        score = SequenceMatcher(None, ex['reference'], ex['hypothesis']).ratio()
                    else:
                        score = ex['word_similarity']
                        
                    length = len(ex['reference'].split())
                    length_scores2[length].append(score)
            
            # Get all lengths that have enough examples
            min_per_length = 3  # Minimum examples per length bucket
            all_lengths = sorted(set(
                length for length, scores in chain(length_scores1.items(), length_scores2.items())
                if len(scores) >= min_per_length
            ))
            
            if not all_lengths:
                print("Warning: Not enough examples per length category")
                return None
            
            # Calculate mean and std for each length
            mean1 = [np.mean(length_scores1.get(l, [])) if length_scores1.get(l, []) else np.nan for l in all_lengths]
            std1 = [np.std(length_scores1.get(l, [])) if len(length_scores1.get(l, [])) > 1 else 0 for l in all_lengths]
            mean2 = [np.mean(length_scores2.get(l, [])) if length_scores2.get(l, []) else np.nan for l in all_lengths]
            std2 = [np.std(length_scores2.get(l, [])) if len(length_scores2.get(l, [])) > 1 else 0 for l in all_lengths]
            
            # Count examples for each length
            counts1 = [len(length_scores1.get(l, [])) for l in all_lengths]
            counts2 = [len(length_scores2.get(l, [])) for l in all_lengths]
            
            # Create visualization
            plt.figure(figsize=(14, 8))
            
            # Plot mean with error bands
            plt.errorbar(all_lengths, mean1, yerr=std1, fmt='o-', capsize=5, 
                        label=self.model1_name, color='royalblue')
            plt.errorbar(all_lengths, mean2, yerr=std2, fmt='s-', capsize=5, 
                        label=self.model2_name, color='darkorange')
            
            # Add count annotations
            for i, (l, m1, m2, c1, c2) in enumerate(zip(all_lengths, mean1, mean2, counts1, counts2)):
                if not np.isnan(m1) and c1 > 0:
                    plt.annotate(f'n={c1}', (l, m1), xytext=(0, 10), 
                                textcoords='offset points', ha='center', fontsize=8)
                if not np.isnan(m2) and c2 > 0:
                    plt.annotate(f'n={c2}', (l, m2), xytext=(0, -20), 
                                textcoords='offset points', ha='center', fontsize=8)
            
            # Add labels and title
            plt.xlabel('Utterance Length (words)')
            plt.ylabel('Word Similarity')
            plt.title('Performance by Utterance Length')
            plt.legend()
            
            # Format axis
            plt.grid(alpha=0.3)
            plt.xticks(all_lengths)
            
            plt.tight_layout()
            
            # Save figure
            output_file = os.path.join(output_dir, 'length_performance_curves.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_file
            
        except Exception as e:
            print(f"Error creating length performance curves: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_comparison_summary(self, output_dir):
        """
        Generate a comparative summary of model performance
        
        Parameters:
        - output_dir: Directory to save summary
        
        Returns:
        - Path to saved summary
        """
        try:
            # Create a summary dictionary
            comparison = {
                'model1_name': self.model1_name,
                'model2_name': self.model2_name,
                'metrics_comparison': {},
                'examples_count': len(self.examples1) if self.examples1 else 0,
                'model1_json': self.json_file1,
                'model2_json': self.json_file2
            }
            
            # Compare all metrics available in both summaries
            for key in set(self.summary1.keys()) & set(self.summary2.keys()):
                # Skip non-numeric values and complex structures
                if isinstance(self.summary1[key], (int, float)) and isinstance(self.summary2[key], (int, float)):
                    val1 = self.summary1[key]
                    val2 = self.summary2[key]
                    diff = val1 - val2
                    percent_diff = diff / val2 * 100 if val2 != 0 else 0
                    
                    comparison['metrics_comparison'][key] = {
                        'model1_value': val1,
                        'model2_value': val2,
                        'difference': diff,
                        'percent_difference': percent_diff
                    }
            
            # Save to JSON file
            output_file = os.path.join(output_dir, 'model_comparison_summary.json')
            with open(output_file, 'w') as f:
                json.dump(comparison, f, indent=2)
            
            # Also create human-readable text summary
            text_file = os.path.join(output_dir, 'model_comparison_summary.txt')
            with open(text_file, 'w') as f:
                f.write(f"=== Model Comparison: {self.model1_name} vs {self.model2_name} ===\n\n")
                
                f.write("Performance Metrics Comparison:\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'Metric':<30} | {self.model1_name:<15} | {self.model2_name:<15} | {'Difference':<15} | {'% Diff':<10}\n")
                f.write("-" * 80 + "\n")
                
                for key, values in comparison['metrics_comparison'].items():
                    better_indicator = ""
                    
                    # Determine which model is better (lower is better for error rates)
                    if key in ['wer', 'cer', 'semantic_wer']:
                        better_indicator = "↑" if values['difference'] > 0 else "↓"
                    else:
                        better_indicator = "↓" if values['difference'] > 0 else "↑"
                    
                    f.write(f"{key:<30} | {values['model1_value']:<15.4f} | {values['model2_value']:<15.4f} | {values['difference']:<15.4f} | {values['percent_difference']:<9.2f}% {better_indicator}\n")
                
                f.write("-" * 80 + "\n\n")
                
                f.write("Summary:\n")
                # Count metrics where each model is better
                model1_better = 0
                model2_better = 0
                
                for key, values in comparison['metrics_comparison'].items():
                    if key in ['wer', 'cer', 'semantic_wer']:
                        if values['difference'] < 0:
                            model1_better += 1
                        elif values['difference'] > 0:
                            model2_better += 1
                    else:
                        if values['difference'] > 0:
                            model1_better += 1
                        elif values['difference'] < 0:
                            model2_better += 1
                
                f.write(f"- {self.model1_name} performs better on {model1_better} metrics\n")
                f.write(f"- {self.model2_name} performs better on {model2_better} metrics\n")
                
                # Add overall conclusion
                if model1_better > model2_better:
                    f.write(f"\nOverall: {self.model1_name} outperforms {self.model2_name} on the majority of metrics.\n")
                elif model2_better > model1_better:
                    f.write(f"\nOverall: {self.model2_name} outperforms {self.model1_name} on the majority of metrics.\n")
                else:
                    f.write(f"\nOverall: {self.model1_name} and {self.model2_name} perform similarly across metrics.\n")
            
            return text_file
            
        except Exception as e:
            print(f"Error generating comparison summary: {e}")
            return None

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
    
    # Add model comparison arguments
    parser.add_argument('--compare', action='store_true', help='Compare two models')
    parser.add_argument('--model1', type=str, help='First model summary.json file')
    parser.add_argument('--model2', type=str, help='Second model summary.json file')
    parser.add_argument('--model1-name', type=str, help='Name for the first model', default=None)
    parser.add_argument('--model2-name', type=str, help='Name for the second model', default=None)
    parser.add_argument('--comparison-output', type=str, help='Output directory for comparison visualizations', default=None)
    
    args = parser.parse_args()
    
    # Load custom phoneme features if specified
    if args.features:
        evaluator = LipReadingEvaluator(load_from_file=args.features)
    
    # Save phoneme features if requested
    if args.save_features:
        evaluator.save_phoneme_features(args.save_features)
    
    # Handle model comparison if requested
    if args.compare:
        if not args.model1 or not args.model2:
            print("Error: Both --model1 and --model2 must be specified for comparison")
            return
        
        # Create comparison output directory if not specified
        if not args.comparison_output:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.comparison_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                             "plots", f"model_comparison_{timestamp}")
        
        # Create model comparator
        print(f"Comparing models: {args.model1} vs {args.model2}")
        comparator = ModelComparator(
            args.model1, 
            args.model2,
            model1_name=args.model1_name,
            model2_name=args.model2_name
        )
        
        # Create comparison visualizations
        comparison_results = comparator.create_comparison_plots(args.comparison_output)
        
        print(f"\nComparison complete. Results saved to: {args.comparison_output}")
        print("Generated visualizations:")
        for viz_type, file_path in comparison_results.items():
            if file_path:
                print(f"- {viz_type}: {os.path.basename(file_path)}")
    
    # Process JSON file if provided
    elif args.json:
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