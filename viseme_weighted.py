#!/usr/bin/env python3
import os
import json
import time
import numpy as np
import math
from collections import Counter, defaultdict
from itertools import chain

# Import from original module
from viseme import LipReadingEvaluator

class WeightedLipReadingEvaluator(LipReadingEvaluator):
    """
    Enhanced lip reading evaluator that uses information-theoretic weights
    for phoneme and viseme similarity calculations
    """
    
    def __init__(self, load_from_file=None, use_weighted_distance=True):
        """
        Initialize with option to use weighted distances
        
        Parameters:
        - load_from_file: Path to file with pre-computed weights (optional)
        - use_weighted_distance: Whether to use information-theoretic weighted distances
        """
        # Initialize parent class
        super().__init__(load_from_file=load_from_file)
        
        # Set weighted distance flag
        self.use_weighted_distance = use_weighted_distance
        
        # Initialize additional caches
        self.viseme_similarity_matrix = {}
        
        # Calculate weights if enabled
        if use_weighted_distance:
            if load_from_file and os.path.exists(load_from_file):
                # Load pre-computed weights from file
                self._load_weights_from_file(load_from_file)
            else:
                # Calculate weights from scratch
                self.feature_weights = self.calculate_information_theoretic_weights()
                
                # Pre-calculate viseme similarity matrix
                self.viseme_similarity_matrix = self.calculate_viseme_similarity_matrix()
    
    def _load_weights_from_file(self, file_path):
        """
        Load pre-computed weights and similarity matrix from file
        
        Parameters:
        - file_path: Path to JSON file with weights
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            self.feature_weights = data.get('feature_weights', {})
            self.viseme_similarity_matrix = {
                tuple(map(int, k.split(','))): v 
                for k, v in data.get('viseme_similarity_matrix', {}).items()
            }
            
            print(f"Loaded weights and similarity matrix from {file_path}")
        except Exception as e:
            print(f"Error loading weights from file: {e}")
            # Initialize with default weights
            self.feature_weights = {}
            self.viseme_similarity_matrix = {}
    
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
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            print(f"Saved weights and similarity matrix to {file_path}")
        except Exception as e:
            print(f"Error saving weights to file: {e}")
    
    def calculate_information_theoretic_weights(self):
        """
        Calculate feature weights based on information theory principles
        using entropy and visual distinctiveness.
        
        Returns:
            dict: Mapping of feature names to weight values (0-1)
        """
        print("Calculating information-theoretic feature weights...")
        weights = {}
        
        # First, make sure we have a reasonable set of phonemes with features
        # If feature cache is empty, populate it with common phonemes
        if not self.phoneme_features_cache:
            common_phonemes = list(self.phoneme_to_viseme.keys())
            for phoneme in common_phonemes:
                if phoneme not in ['.', ' ', '-']:
                    self.get_phoneme_features(phoneme)
        
        phoneme_inventory = list(self.phoneme_features_cache.keys())
        print(f"Using {len(phoneme_inventory)} phonemes in inventory")
        
        # Get all features across the inventory
        all_features = set()
        for features in self.phoneme_features_cache.values():
            all_features.update(features.keys())
        
        # Remove special features that aren't standard phonetic features
        all_features = {f for f in all_features if f not in 
                        ['is_silence', 'unknown', 'error', 'is_vowel', 'is_consonant']}
        
        print(f"Analyzing {len(all_features)} phonetic features")
        
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
            
            # Using numpy for stability in log calculations
            entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
            
            # Normalize by maximum possible entropy
            max_entropy = np.log2(len(value_counts)) if len(value_counts) > 1 else 1
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
                viseme = self.phoneme_to_viseme.get(phoneme, -1)  # Use -1 for unknown
                
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
        print("\nInformation-theoretic feature weights:")
        print("Feature | Entropy | Visual Distinctiveness | Final Weight")
        print("--------|---------|------------------------|-------------")
        for feature in sorted(weights.keys(), key=lambda f: weights[f], reverse=True):
            entropy = feature_entropies.get(feature, 0)
            distinctiveness = visual_distinctiveness.get(feature, 0)
            weight = weights[feature]
            print(f"{feature:8} | {entropy:.3f}  | {distinctiveness:.3f}                | {weight:.3f}")
        
        return weights
    
    def _calculate_weighted_feature_distance(self, features1, features2):
        """
        Calculate weighted distance between two phonemes using feature weights
        
        Args:
            features1: Feature dictionary for first phoneme
            features2: Feature dictionary for second phoneme
            
        Returns:
            float: Weighted distance between 0.0 (identical) and 1.0 (different)
        """
        if not hasattr(self, 'feature_weights') or not self.feature_weights:
            return 1.0  # Maximum distance if no weights
        
        total_diff = 0.0
        total_weight = 0.0
        
        # Common binary phonological features to compare
        binary_features = [
            'syllabic', 'consonantal', 'sonorant', 'continuant',
            'voice', 'nasal', 'strident', 'lateral',
            'labial', 'coronal', 'dorsal', 'high', 'low', 'back', 'round'
        ]
        
        # Compare each feature that exists in both phonemes
        for feature in binary_features:
            if feature in features1 and feature in features2 and feature in self.feature_weights:
                # Get feature values (ensuring they're numeric)
                val1 = 1 if features1[feature] else 0
                val2 = 1 if features2[feature] else 0
                
                # Calculate difference and apply weight
                diff = abs(val1 - val2)
                weight = self.feature_weights[feature]
                
                total_diff += diff * weight
                total_weight += weight
        
        # Calculate weighted average difference
        if total_weight > 0:
            weighted_distance = total_diff / total_weight
            return min(1.0, weighted_distance)  # Ensure it's within range
        else:
            return 1.0  # Maximum distance if no common features
    
    def calculate_viseme_similarity_matrix(self):
        """
        Pre-calculate similarity between all viseme pairs based on
        phonetic features and information-theoretic weights.
        
        Returns:
            dict: Mapping of (viseme1, viseme2) tuples to similarity values
        """
        print("Calculating viseme similarity matrix...")
        similarity_matrix = {}
        
        # Get all viseme classes
        viseme_classes = sorted(set(self.phoneme_to_viseme.values()))
        print(f"Processing {len(viseme_classes)} viseme classes")
        
        # Get representative phonemes for each viseme class
        viseme_to_phonemes = {}
        for viseme in viseme_classes:
            phonemes = [p for p, v in self.phoneme_to_viseme.items() if v == viseme]
            # Limit to a reasonable number of representatives
            viseme_to_phonemes[viseme] = phonemes[:5]
        
        # Calculate similarity between each pair of viseme classes
        for i, viseme1 in enumerate(viseme_classes):
            for viseme2 in viseme_classes:
                # Skip if already calculated (symmetrical)
                if (viseme1, viseme2) in similarity_matrix or (viseme2, viseme1) in similarity_matrix:
                    continue
                
                # Get representative phonemes
                phonemes1 = viseme_to_phonemes[viseme1]
                phonemes2 = viseme_to_phonemes[viseme2]
                
                if not phonemes1 or not phonemes2:
                    # If no phonemes for either viseme, set maximum distance
                    similarity = 0.0
                else:
                    # Calculate pairwise distances between representative phonemes
                    distances = []
                    for p1 in phonemes1:
                        for p2 in phonemes2:
                            try:
                                # Use calculated distance
                                distance = self.calculate_phonetic_distance(p1, p2)
                                distances.append(distance)
                            except Exception as e:
                                print(f"Error comparing '{p1}' and '{p2}': {e}")
                    
                    # Calculate average distance
                    if distances:
                        avg_distance = sum(distances) / len(distances)
                        # Convert distance to similarity (1 - distance)
                        similarity = 1.0 - avg_distance
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
        
        return similarity_matrix
    
    def calculate_phonetic_distance(self, phoneme1, phoneme2):
        """
        Calculate phonetic distance between two phonemes using either
        panphon's distance metrics or information-theoretic weighted distance.
        
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
        
        # If information-theoretic weights are enabled and available
        if self.use_weighted_distance and hasattr(self, 'feature_weights') and self.feature_weights:
            try:
                # Get feature vectors for both phonemes
                features1 = self.get_phoneme_features(phoneme1)
                features2 = self.get_phoneme_features(phoneme2)
                
                # Calculate weighted feature distance
                distance = self._calculate_weighted_feature_distance(features1, features2)
                
                # Cache and return
                self.similarity_cache[cache_key] = distance
                return distance
            except Exception as e:
                print(f"Warning: Error calculating weighted distance between '{phoneme1}' and '{phoneme2}': {e}")
                # Fall back to standard panphon distance
                pass
        
        # Use standard panphon distance if weights not available or error occurred
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
    
    def calculate_viseme_alignment(self, seq1, seq2):
        """
        Calculate alignment between two viseme sequences using dynamic programming
        with weighted costs based on viseme similarity.
        
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
                    # Exact match
                    cost = 0
                elif self.use_weighted_distance and self.viseme_similarity_matrix:
                    # Use weighted cost based on viseme similarity
                    # Convert similarity to distance (1 - similarity)
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
                elif self.use_weighted_distance and self.viseme_similarity_matrix:
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
    
    def evaluate_pair_with_details(self, reference, hypothesis):
        """
        Evaluate a single reference-hypothesis pair with detailed metrics using both
        standard and weighted approaches for comparison.
        
        Parameters:
        - reference: Reference text (ground truth)
        - hypothesis: Hypothesis text (predicted)
        
        Returns:
        - Dictionary with evaluation results for both approaches
        """
        start_time = time.time()
        
        # Store original weighted distance setting
        original_setting = self.use_weighted_distance
        
        try:
            # Evaluate with standard distance (no weights)
            self.use_weighted_distance = False
            standard_results = self.evaluate_pair(reference, hypothesis)
            
            # Evaluate with weighted distance
            self.use_weighted_distance = True
            weighted_results = self.evaluate_pair(reference, hypothesis)
            
            # Restore original setting
            self.use_weighted_distance = original_setting
            
            # Combine results
            combined_results = {
                'reference': reference,
                'hypothesis': hypothesis,
                'phonemes': standard_results.get('ref_phonemes', []),
                'standard': {
                    'phonetic_edit_distance': standard_results.get('phonetic_edit_distance', float('inf')),
                    'viseme_edit_distance': standard_results.get('viseme_edit_distance', float('inf')),
                    'viseme_alignment_score': standard_results.get('viseme_alignment_score', 0.0),
                },
                'weighted': {
                    'phonetic_edit_distance': weighted_results.get('phonetic_edit_distance', float('inf')),
                    'viseme_edit_distance': weighted_results.get('viseme_edit_distance', float('inf')),
                    'viseme_alignment_score': weighted_results.get('viseme_alignment_score', 0.0),
                },
                'processing_time': time.time() - start_time
            }
            
            return combined_results
            
        except Exception as e:
            print(f"ERROR in evaluate_pair_with_details: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Restore original setting
            self.use_weighted_distance = original_setting
            
            # Return a minimal result
            return {
                'reference': reference,
                'hypothesis': hypothesis,
                'error': str(e),
                'standard': {'viseme_alignment_score': 0.0},
                'weighted': {'viseme_alignment_score': 0.0}
            }
    
    def analyze_json_dataset_with_comparisons(self, json_file, output_file=None, max_samples=None):
        """
        Analyze a dataset with both standard and weighted approaches for comparison
        
        Parameters:
        - json_file: Path to JSON file with reference-hypothesis pairs
        - output_file: Path to save analysis results (optional)
        - max_samples: Maximum number of samples to process (default: all)
        
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
                results = self.evaluate_pair_with_details(ref, hyp)
                all_results.append(results)
            
            # Calculate summary statistics
            summary = {
                'num_examples': len(all_results),
                'standard': {
                    'avg_viseme_score': np.mean([r['standard']['viseme_alignment_score'] for r in all_results]),
                    'avg_phonetic_distance': np.mean([r['standard']['phonetic_edit_distance'] for r in all_results 
                                                    if r['standard']['phonetic_edit_distance'] != float('inf')]),
                },
                'weighted': {
                    'avg_viseme_score': np.mean([r['weighted']['viseme_alignment_score'] for r in all_results]),
                    'avg_phonetic_distance': np.mean([r['weighted']['phonetic_edit_distance'] for r in all_results
                                                    if r['weighted']['phonetic_edit_distance'] != float('inf')]),
                }
            }
            
            # Calculate difference statistics
            score_diffs = [r['weighted']['viseme_alignment_score'] - r['standard']['viseme_alignment_score'] 
                          for r in all_results]
            
            summary['comparison'] = {
                'avg_score_difference': np.mean(score_diffs),
                'max_score_improvement': max(score_diffs),
                'percent_improved': sum(1 for d in score_diffs if d > 0) / len(score_diffs) * 100,
                'percent_unchanged': sum(1 for d in score_diffs if d == 0) / len(score_diffs) * 100,
                'percent_worse': sum(1 for d in score_diffs if d < 0) / len(score_diffs) * 100,
            }
            
            # Print summary
            print("\n=== ANALYSIS SUMMARY ===")
            print(f"Total examples: {summary['num_examples']}")
            print("\nStandard approach:")
            print(f"  Average viseme score: {summary['standard']['avg_viseme_score']:.3f}")
            print(f"  Average phonetic distance: {summary['standard']['avg_phonetic_distance']:.3f}")
            print("\nWeighted approach:")
            print(f"  Average viseme score: {summary['weighted']['avg_viseme_score']:.3f}")
            print(f"  Average phonetic distance: {summary['weighted']['avg_phonetic_distance']:.3f}")
            print("\nComparison:")
            print(f"  Average score difference: {summary['comparison']['avg_score_difference']:.3f}")
            print(f"  Maximum score improvement: {summary['comparison']['max_score_improvement']:.3f}")
            print(f"  Examples improved: {summary['comparison']['percent_improved']:.1f}%")
            print(f"  Examples unchanged: {summary['comparison']['percent_unchanged']:.1f}%")
            print(f"  Examples worse: {summary['comparison']['percent_worse']:.1f}%")
            
            # Save results if output file specified
            if output_file:
                results_to_save = {
                    'summary': summary,
                    'examples': all_results
                }
                
                with open(output_file, 'w') as f:
                    json.dump(results_to_save, f, indent=2)
                
                print(f"\nSaved analysis results to {output_file}")
            
            return {
                'summary': summary,
                'results': all_results
            }
            
        except Exception as e:
            print(f"Error analyzing dataset: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Main function for demonstrating and comparing different alignment approaches"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Viseme Alignment with Information-Theoretic Weights')
    parser.add_argument('--json', type=str, help='JSON file with reference-hypothesis pairs')
    parser.add_argument('--weights', type=str, help='JSON file with pre-computed weights (optional)')
    parser.add_argument('--output', type=str, help='Output file for analysis results')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to process')
    parser.add_argument('--compare', action='store_true', help='Run comparison examples')
    args = parser.parse_args()
    
    # Create weighted evaluator
    evaluator = WeightedLipReadingEvaluator(load_from_file=args.weights)
    
    # Save weights if computed
    if args.weights is None and evaluator.feature_weights:
        weights_file = 'viseme_weights.json'
        evaluator.save_weights_to_file(weights_file)
        print(f"Saved computed weights to {weights_file}")
    
    # Run comparison examples if requested
    if args.compare:
        example_pairs = [
            ("Hello world", "Hello world"),  # Perfect match
            ("Hello world", "Hello word"),   # Small difference
            ("The cat sat on the mat", "The cat sat on a mat"),  # Article change
            ("Please pass the salt", "Please pass the fault"),   # Visually similar sounds
            ("Turn left at the corner", "Turn right at the corner"),  # Visually different
        ]
        
        print("\n=== Comparing Standard vs. Weighted Viseme Alignment ===\n")
        
        for ref, hyp in example_pairs:
            print(f"\nReference: '{ref}'")
            print(f"Hypothesis: '{hyp}'")
            
            # Evaluate with both approaches
            results = evaluator.evaluate_pair_with_details(ref, hyp)
            
            # Print results
            print(f"Standard score: {results['standard']['viseme_alignment_score']:.3f}")
            print(f"Weighted score: {results['weighted']['viseme_alignment_score']:.3f}")
            diff = results['weighted']['viseme_alignment_score'] - results['standard']['viseme_alignment_score']
            print(f"Difference: {diff:.3f} ({'better' if diff > 0 else 'worse' if diff < 0 else 'same'})")
    
    # Process JSON file if provided
    if args.json:
        results = evaluator.analyze_json_dataset_with_comparisons(
            args.json, 
            output_file=args.output,
            max_samples=args.max_samples
        )
        
        if results:
            print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
