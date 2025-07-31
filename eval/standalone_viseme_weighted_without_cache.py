#!/usr/bin/env python3
import os
import json
import time
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
import csv

class LipReadingEvaluator:
    """Comprehensive lip reading evaluation system with information-theoretic weights"""
    
    def __init__(self, use_ipa=False):
        """Initialize the evaluator with phoneme mappings and converters
        
        Parameters:
        - use_ipa: Whether to use direct IPA conversion instead of G2P
        """
        
        # Track whether to use direct IPA conversion or G2P
        self.use_ipa = use_ipa
        
        # Import essential libraries
        if not use_ipa:
            try:
                import g2p_en
                self.g2p = g2p_en.G2p()
            except ImportError:
                print("ERROR: g2p_en library is required for phoneme conversion")
                print("Install with: pip install g2p_en")
                raise
        else:
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
                print("ERROR: phonemizer library is required for direct IPA conversion")
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
        # Handle special characters
        if phoneme in ['.', ' ', '-'] or not phoneme.strip():
            return {'is_silence': 1}
            
        # Try to get panphon features
        try:
            # Get feature vector from panphon
            feature_vector = self.ft.word_to_vector_list(phoneme, numeric=True)
            
            if not feature_vector:
                # If panphon couldn't process it, create minimal features
                return {'unknown': 1}
                
            # Convert to dictionary with feature names
            feature_dict = {}
            for i, feature_name in enumerate(self.ft.names):
                feature_dict[feature_name] = int(feature_vector[0][i])
                
            # Add derived features for convenience
            feature_dict['is_vowel'] = feature_dict.get('syl', 0) == 1
            feature_dict['is_consonant'] = feature_dict.get('syl', 0) == 0
            
            return feature_dict
            
        except Exception as e:
            print(f"Warning: Error getting features for '{phoneme}': {e}")
            # Create minimal features if panphon fails
            return {'error': 1}

    def calculate_phonetic_distance(self, phoneme1, phoneme2):
        """
        Calculate phonetic distance between two phonemes using panphon's distance metrics.
        
        Args:
            phoneme1: First phoneme
            phoneme2: Second phoneme
            
        Returns:
            float: Distance value between 0.0 (identical) and 1.0 (maximally different)
        """
        # Special case: if either is silence or special character
        if phoneme1 in ['.', ' ', '-'] or phoneme2 in ['.', ' ', '-']:
            # Silence compared to anything is maximally different
            return 1.0
            
        # Calculate distance using panphon
        try:
            # Calculate weighted feature edit distance
            distance = self.dst.feature_edit_distance(phoneme1, phoneme2)
            
            # Normalize to 0-1 range (panphon distances can be larger)
            return min(1.0, distance / 10.0)
            
        except Exception as e:
            print(f"Warning: Error calculating distance between '{phoneme1}' and '{phoneme2}': {e}")
            # Default to maximum distance on error
            return 1.0

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
        """Convert English text to phoneme sequence"""
        # Use either direct IPA conversion or G2P based on initialization setting
        if self.use_ipa:
            return self.text_to_phonemes_ipa(text)
        else:
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
            
            return processed_phonemes
    
    def text_to_phonemes_ipa(self, text):
        """Convert English text directly to IPA phoneme sequence using phonemizer"""
        # Use the backend directly for processing
        raw_phonemes = self.phonemizer_backend.phonemize(
            [text], 
            separator=self.separator
        )[0]
        
        # Process the phonemes - split by spaces and clean up
        processed_phonemes = []
        
        # Extract diphthongs directly from our mapping
        diphthongs = [k for k in self.phoneme_to_viseme.keys() 
                     if len(k) > 1 and all(c.isalpha() for c in k)]
        
        # Focus on key diphthongs for viseme classes 9, 10, 11
        key_diphthongs = ['ɔɪ', 'aʊ', 'aɪ']
        
        for phoneme in raw_phonemes.split():
            # Remove any stress markers or other non-essential symbols
            cleaned = phoneme.strip()
            
            # Check if the whole phoneme is already in our mapping
            if cleaned in self.phoneme_to_viseme:
                processed_phonemes.append(cleaned)
                continue
            
            # Process character by character with priority to diphthongs
            i = 0
            while i < len(cleaned):
                # Special explicit check for 'ɔɪ' diphthong (viseme 10)
                if i < len(cleaned) - 1 and cleaned[i] == 'ɔ' and cleaned[i+1] == 'ɪ':
                    processed_phonemes.append('ɔɪ')  # Explicitly add the diphthong
                    i += 2
                    continue
                    
                # Check for other diphthongs
                if i < len(cleaned) - 1:
                    dipth = cleaned[i:i+2]
                    if dipth in diphthongs:
                        processed_phonemes.append(dipth)
                        i += 2
                        continue
                
                # If no diphthong found, add the individual character
                processed_phonemes.append(cleaned[i])
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
        start_time = time.time()
        
        # Convert texts to phoneme sequences
        try:
            ref_phonemes = self.text_to_phonemes(reference)
            hyp_phonemes = self.text_to_phonemes(hypothesis)        
            # Calculate phonetic alignment and edit distance
            align_start = time.time()
            alignment, edit_distance = self.calculate_phonetic_alignment(ref_phonemes, hyp_phonemes)
            
            # Convert phonemes to visemes using most appropriate mapping
            ref_visemes = [self.get_closest_viseme(p) for p in ref_phonemes]
            hyp_visemes = [self.get_closest_viseme(p) for p in hyp_phonemes]
            
            # Calculate viseme-level alignment and score
            vis_start = time.time()
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

# Enhanced lip reading evaluator with weighted measurements
class WeightedLipReadingEvaluator(LipReadingEvaluator):
    """
    Enhanced lip reading evaluator that uses information-theoretic weights
    for phoneme and viseme similarity calculations
    """
    
    def __init__(self, use_weighted_distance=True, use_ipa=False):
        """
        Initialize with option to use weighted distances
        
        Parameters:
        - use_weighted_distance: Whether to use information-theoretic weighted distances
        - use_ipa: Whether to use direct IPA conversion instead of G2P
        """
        # Initialize parent class
        super().__init__(use_ipa=use_ipa)
        
        # Set weighted distance flag
        self.use_weighted_distance = use_weighted_distance
        
        # Initialize similarity matrix
        self.viseme_similarity_matrix = {}
        
        # Calculate weights if enabled
        if use_weighted_distance:
            # Calculate weights from scratch
            self.feature_weights = self.calculate_information_theoretic_weights()
            
            # Pre-calculate viseme similarity matrix
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
                'viseme_similarity_matrix': serializable_matrix,
                'conversion_method': 'ipa' if self.use_ipa else 'g2p'  # Add method info
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
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
        
        # Generate a phoneme inventory based on current conversion method
        if self.use_ipa:
            print("Using IPA-based phoneme inventory for weight calculation")
            # Create a suitable phoneme inventory using IPA directly
            # Sample some common English sentences and extract phonemes
            sample_sentences = [
                "Hello world",
                "The quick brown fox jumps over the lazy dog",
                "How are you today",
                "Please speak clearly",
                "What time is it",
                "Thank you very much",
                "Can you help me please",
                "I would like to buy a ticket",
                "Where is the nearest train station",
                "Good morning, good afternoon, good evening"
            ]
            
            # Extract phonemes from these sentences using the IPA converter
            phoneme_inventory = set()
            for sentence in sample_sentences:
                phonemes = self.text_to_phonemes(sentence)
                phoneme_inventory.update(phonemes)
            
            # Add phonemes directly from our mapping as well
            phoneme_inventory.update(self.phoneme_to_viseme.keys())
            
            print(f"Generated {len(phoneme_inventory)} phonemes for IPA-based inventory")
        else:
            # Use the standard phoneme inventory from our mapping
            phoneme_inventory = list(self.phoneme_to_viseme.keys())
        
        # Remove silence markers
        phoneme_inventory = [p for p in phoneme_inventory if p not in ['.', ' ', '-', '']]
        
        print(f"Using {len(phoneme_inventory)} phonemes in inventory for weight calculation")
        
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
                    viseme = self.phoneme_to_viseme.get(phoneme, -1)  # Use -1 for unknown
                    
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
    
    def _calculate_weighted_feature_distance(self, phoneme1, phoneme2):
        """
        Calculate weighted distance between phonemes using panphon's feature vectors
        and information-theoretic weights
        
        Args:
            phoneme1: First phoneme
            phoneme2: Second phoneme
            
        Returns:
            float: Weighted distance between 0.0 (identical) and 1.0 (different)
        """
        if not hasattr(self, 'feature_weights') or not self.feature_weights:
            return 1.0  # Maximum distance if no weights
        
        # Get raw feature vectors from panphon
        fv1 = self.ft.word_to_vector_list(phoneme1, numeric=True)
        fv2 = self.ft.word_to_vector_list(phoneme2, numeric=True)
        
        if not fv1 or not fv2:
            return 1.0  # Maximum distance if features can't be computed
        
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
            phonemes = [p for p, v in self.phoneme_to_viseme.items() if v == viseme and p not in ['.', ' ', '-']]
            # Limit to a reasonable number of representatives, but ensure diversity
            if len(phonemes) > 5:
                # Try to select diverse phonemes by taking from different phonetic categories
                vowels = [p for p in phonemes if self.get_phoneme_features(p).get('is_vowel', False)]
                consonants = [p for p in phonemes if self.get_phoneme_features(p).get('is_consonant', False)]
                selected = []
                
                # Take a mix of vowels and consonants if available
                if vowels:
                    selected.extend(vowels[:3 if len(vowels) > 3 else len(vowels)])
                if consonants:
                    selected.extend(consonants[:5-len(selected) if len(consonants) > (5-len(selected)) else len(consonants)])
                
                # If we didn't get enough, add any remaining phonemes
                if len(selected) < 5:
                    remaining = [p for p in phonemes if p not in selected]
                    selected.extend(remaining[:5-len(selected)])
                
                viseme_to_phonemes[viseme] = selected
            else:
                viseme_to_phonemes[viseme] = phonemes
        
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
                    # Calculate pairwise distances between representative phonemes
                    distances = []
                    for p1 in phonemes1:
                        for p2 in phonemes2:
                            try:
                                # Use calculated distance with panphon features
                                distance = self.calculate_phonetic_distance(p1, p2)
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
        Calculate phonetic distance between two phonemes using either
        panphon's distance metrics or information-theoretic weighted distance.
        
        Args:
            phoneme1: First phoneme
            phoneme2: Second phoneme
            
        Returns:
            float: Distance value between 0.0 (identical) and 1.0 (maximally different)
        """
        # Special case: if either is silence or special character
        if phoneme1 in ['.', ' ', '-'] or phoneme2 in ['.', ' ', '-']:
            # Silence compared to anything is maximally different
            return 1.0
        
        # If information-theoretic weights are enabled and available
        if self.use_weighted_distance and hasattr(self, 'feature_weights') and self.feature_weights:
            try:
                # Calculate weighted feature distance directly using phonemes
                return self._calculate_weighted_feature_distance(phoneme1, phoneme2)
            except Exception as e:
                print(f"Warning: Error calculating weighted distance between '{phoneme1}' and '{phoneme2}': {e}")
                # Fall back to standard panphon distance
                pass
        
        # Use standard panphon distance if weights not available or error occurred
        try:
            # Use panphon's feature edit distance
            distance = self.dst.feature_edit_distance(phoneme1, phoneme2)
            
            # Normalize to 0-1 range (panphon distances can be larger)
            return min(1.0, distance / 10.0)
        except Exception as e:
            print(f"Warning: Error calculating distance between '{phoneme1}' and '{phoneme2}': {e}")
            # Default to maximum distance on error
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

    def calculate_weighted_phonetic_edit_distance(self, phoneme_seq1, phoneme_seq2):
        """
        Calculate phonetic edit distance between two phoneme sequences using weighted distance.
        
        Args:
            phoneme_seq1: First phoneme sequence
            phoneme_seq2: Second phoneme sequence
            
        Returns:
            float: Weighted edit distance value
        """
        # Initialize matrix
        m, n = len(phoneme_seq1), len(phoneme_seq2)
        dp = [[0.0] * (n + 1) for _ in range(m + 1)]
        
        # Fill first row and column
        for i in range(m + 1):
            dp[i][0] = float(i)
        for j in range(n + 1):
            dp[0][j] = float(j)
        
        # Fill the matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if phoneme_seq1[i-1] == phoneme_seq2[j-1]:
                    # Exact match
                    cost = 0.0
                else:
                    # Use weighted phonetic distance
                    cost = self.calculate_phonetic_distance(phoneme_seq1[i-1], phoneme_seq2[j-1])
                
                dp[i][j] = min(
                    dp[i-1][j] + 1.0,        # Deletion
                    dp[i][j-1] + 1.0,        # Insertion
                    dp[i-1][j-1] + cost      # Substitution
                )
        
        return dp[m][n]
        
    def evaluate_pair(self, reference, hypothesis):
        """
        Override parent method to evaluate a reference-hypothesis pair using weighted distance
        
        Args:
            reference: Reference text (ground truth)
            hypothesis: Hypothesis text (predicted)
            
        Returns:
            dict: Evaluation results
        """
        # Convert text to phonemes
        ref_phonemes = self.text_to_phonemes(reference)
        hyp_phonemes = self.text_to_phonemes(hypothesis)
        
        # Convert phonemes to visemes
        ref_visemes = [self.phoneme_to_viseme.get(p, -1) for p in ref_phonemes]
        hyp_visemes = [self.phoneme_to_viseme.get(p, -1) for p in hyp_phonemes]
        
        # Calculate phonetic edit distance - use weighted or standard approach
        if self.use_weighted_distance:
            phonetic_distance = self.calculate_weighted_phonetic_edit_distance(ref_phonemes, hyp_phonemes)
            phonetic_alignment = None  # We don't need the alignment for the weighted version
        else:
            # Use parent class's method for standard distance
            phonetic_alignment, phonetic_distance = super().calculate_phonetic_alignment(ref_phonemes, hyp_phonemes)
        
        # Calculate viseme alignment
        alignment, edit_distance = self.calculate_viseme_alignment(ref_visemes, hyp_visemes)
        
        # Calculate normalized scores (0-1, higher is better)
        max_viseme_len = max(len(ref_visemes), len(hyp_visemes))
        max_phoneme_len = max(len(ref_phonemes), len(hyp_phonemes))
        
        viseme_alignment_score = 1.0 - (edit_distance / max_viseme_len if max_viseme_len > 0 else 0.0)
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
            'viseme_alignment_score': viseme_alignment_score
        }
        
        return results
    
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
        
        # Store original weighted distance setting (but preserve IPA setting)
        original_setting = self.use_weighted_distance
        # Store current IPA setting
        current_ipa_setting = self.use_ipa
        
        try:
            # Evaluate with standard distance (no weights)
            self.use_weighted_distance = False
            # Keep IPA setting consistent
            standard_results = self.evaluate_pair(reference, hypothesis)
            
            # Evaluate with weighted distance
            self.use_weighted_distance = True
            # Keep IPA setting consistent
            weighted_results = self.evaluate_pair(reference, hypothesis)
            
            # Restore original setting
            self.use_weighted_distance = original_setting
            # No need to restore use_ipa since we never changed it
            
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
                    'viseme_alignment_score': standard_results.get('viseme_alignment_score', 0.0),
                },
                'weighted': {
                    'phonetic_edit_distance': weighted_results.get('phonetic_edit_distance', float('inf')),
                    'phonetic_alignment_score': weighted_results.get('phonetic_alignment_score', 0.0),
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
            # No need to restore use_ipa since we never changed it
            
            # Return a minimal result
            return {
                'reference': reference,
                'hypothesis': hypothesis,
                'error': str(e),
                'standard': {'viseme_alignment_score': 0.0},
                'weighted': {'viseme_alignment_score': 0.0}
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
                results = self.evaluate_pair_with_details(ref, hyp)
                all_results.append(results)
            
            # Calculate summary statistics
            summary = {
                'num_examples': len(all_results),
                'standard': {
                    'avg_viseme_score': np.mean([r['standard']['viseme_alignment_score'] for r in all_results]),
                    'avg_phonetic_distance': np.mean([r['standard']['phonetic_edit_distance'] for r in all_results 
                                                    if r['standard']['phonetic_edit_distance'] != float('inf')]),
                    'avg_phonetic_score': np.mean([r['standard']['phonetic_alignment_score'] for r in all_results]),
                },
                'weighted': {
                    'avg_viseme_score': np.mean([r['weighted']['viseme_alignment_score'] for r in all_results]),
                    'avg_phonetic_distance': np.mean([r['weighted']['phonetic_edit_distance'] for r in all_results
                                                    if r['weighted']['phonetic_edit_distance'] != float('inf')]),
                    'avg_phonetic_score': np.mean([r['weighted']['phonetic_alignment_score'] for r in all_results]),
                }
            }
            
            # Calculate difference statistics
            viseme_score_diffs = [r['weighted']['viseme_alignment_score'] - r['standard']['viseme_alignment_score'] 
                          for r in all_results]
            phonetic_score_diffs = [r['weighted']['phonetic_alignment_score'] - r['standard']['phonetic_alignment_score'] 
                          for r in all_results]
            
            summary['comparison'] = {
                'avg_viseme_score_difference': np.mean(viseme_score_diffs),
                'max_viseme_score_improvement': max(viseme_score_diffs),
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
            print(f"  Average viseme score: {summary['weighted']['avg_viseme_score']:.3f}")
            print(f"  Average phonetic score: {summary['weighted']['avg_phonetic_score']:.3f}")
            print(f"  Average phonetic distance: {summary['weighted']['avg_phonetic_distance']:.3f}")
            print("\nComparison:")
            print(f"  Average viseme score difference: {summary['comparison']['avg_viseme_score_difference']:.3f}")
            print(f"  Maximum viseme score improvement: {summary['comparison']['max_viseme_score_improvement']:.3f}")
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
                'wgt_viseme_score',
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
                    row['std_viseme_score'] = round(std.get('viseme_alignment_score', 0), 4) if 'viseme_alignment_score' in std else ''
                    
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
                    row['wgt_viseme_score'] = round(wgt.get('viseme_alignment_score', 0), 4) if 'viseme_alignment_score' in wgt else ''
                    
                    # Handle special case for infinity
                    wgt_ped = wgt.get('phonetic_edit_distance', None)
                    if wgt_ped == float('inf'):
                        row['wgt_phonetic_distance'] = 'inf'
                    elif wgt_ped is not None:
                        row['wgt_phonetic_distance'] = round(wgt_ped, 4)
                    else:
                        row['wgt_phonetic_distance'] = ''
                    
                    # Calculate score difference
                    if 'viseme_alignment_score' in std and 'viseme_alignment_score' in wgt:
                        row['score_difference'] = round(wgt['viseme_alignment_score'] - std['viseme_alignment_score'], 4)
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
            ref_visemes = [self.phoneme_to_viseme.get(p, -1) for p in ref_phonemes]
            hyp_visemes = [self.phoneme_to_viseme.get(p, -1) for p in hyp_phonemes]
            
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
        all_visemes = sorted(set(self.phoneme_to_viseme.values()))
        
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
            
            # Save matrix data to file in the output directory for analysis
            matrix_data = {
                'standard_matrix': std_cm_norm.tolist(),
                'weighted_matrix': wgt_cm_norm.tolist(),
                'difference_matrix': (wgt_cm_norm - std_cm_norm).tolist(),
                'viseme_labels': all_visemes,
                'viseme_names': {str(k): v for k, v in viseme_names.items()},
                'conversion_method': 'ipa' if self.use_ipa else 'g2p'
            }
            matrix_file = os.path.join(output_dir, 'confusion_matrices.json')
            with open(matrix_file, 'w') as f:
                json.dump(matrix_data, f, indent=2)
            print(f"Saved confusion matrix data to {matrix_file}")
            
            # Better color maps - both blue now
            # Use a blue-white colormap for both matrices
            std_cmap = plt.cm.Blues
            wgt_cmap = plt.cm.Blues
            # Use a balanced diverging colormap for the difference matrix
            diff_cmap = sns.diverging_palette(240, 10, as_cmap=True)
            
            # Create figure with enough space for the legend
            plt.figure(figsize=(16, 12))
            
            # Split figure for main plot and legend
            gs = plt.GridSpec(1, 2, width_ratios=[3, 1])
            
            # Main heatmap axis
            ax_heatmap = plt.subplot(gs[0])
            ax_legend = plt.subplot(gs[1])
            
            # Standard confusion matrix
            sns.heatmap(std_cm_norm, annot=True, fmt='.2f', cmap=std_cmap, square=True, vmin=0, vmax=1, 
                        annot_kws={"size": 8}, xticklabels=viseme_labels, yticklabels=viseme_labels,
                        ax=ax_heatmap)
            ax_heatmap.set_title('Standard Approach - Viseme Confusion Matrix')
            ax_heatmap.set_xlabel('Predicted Viseme')
            ax_heatmap.set_ylabel('True Viseme')
            
            # Simple legend in the right panel
            ax_legend.axis('off')  # Turn off axis for legend panel
            legend_text = "Viseme Reference:\n\n"
            for v in all_visemes:
                legend_text += f"{v}: {viseme_names.get(v, 'Unknown')}\n"
            
            ax_legend.text(0, 0.5, legend_text, va='center', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/standard_confusion_matrix.png", dpi=300)
            plt.close()
            
            # Weighted confusion matrix
            plt.figure(figsize=(16, 12))
            
            # Split figure for main plot and legend
            gs = plt.GridSpec(1, 2, width_ratios=[3, 1])
            
            # Main heatmap axis
            ax_heatmap = plt.subplot(gs[0])
            ax_legend = plt.subplot(gs[1])
            
            sns.heatmap(wgt_cm_norm, annot=True, fmt='.2f', cmap=wgt_cmap, square=True, vmin=0, vmax=1, 
                        annot_kws={"size": 8}, xticklabels=viseme_labels, yticklabels=viseme_labels,
                        ax=ax_heatmap)
            ax_heatmap.set_title('Weighted Approach - Viseme Confusion Matrix')
            ax_heatmap.set_xlabel('Predicted Viseme')
            ax_heatmap.set_ylabel('True Viseme')
            
            # Simple legend in the right panel
            ax_legend.axis('off')  # Turn off axis for legend panel
            legend_text = "Viseme Reference:\n\n"
            for v in all_visemes:
                legend_text += f"{v}: {viseme_names.get(v, 'Unknown')}\n"
            
            ax_legend.text(0, 0.5, legend_text, va='center', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/weighted_confusion_matrix.png", dpi=300)
            plt.close()
            
            # Difference matrix
            plt.figure(figsize=(16, 12))
            
            # Split figure for main plot and legend
            gs = plt.GridSpec(1, 2, width_ratios=[3, 1])
            
            # Main heatmap axis
            ax_heatmap = plt.subplot(gs[0])
            ax_legend = plt.subplot(gs[1])
            
            diff_cm = wgt_cm_norm - std_cm_norm
            sns.heatmap(diff_cm, annot=True, fmt='.2f', cmap=diff_cmap, square=True, center=0, vmin=-0.5, vmax=0.5, 
                        annot_kws={"size": 8}, xticklabels=viseme_labels, yticklabels=viseme_labels,
                        ax=ax_heatmap)
            ax_heatmap.set_title('Difference (Weighted - Standard) Confusion Matrix')
            ax_heatmap.set_xlabel('Predicted Viseme')
            ax_heatmap.set_ylabel('True Viseme')
            
            # Simple legend in the right panel
            ax_legend.axis('off')  # Turn off axis for legend panel
            legend_text = "Viseme Reference:\n\n"
            for v in all_visemes:
                legend_text += f"{v}: {viseme_names.get(v, 'Unknown')}\n"
            
            ax_legend.text(0, 0.5, legend_text, va='center', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/difference_confusion_matrix.png", dpi=300)
            plt.close()
    
def main():
    """Main function for demonstrating and comparing different alignment approaches"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Viseme Alignment with Information-Theoretic Weights')
    parser.add_argument('--json', type=str, help='JSON file with reference-hypothesis pairs')
    parser.add_argument('--weights', type=str, help='Path to save computed weights (will not load from this file)')
    parser.add_argument('--save_dir', type=str, default='viseme_output', help='Directory to save all outputs (results, plots, weights)')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to process')
    parser.add_argument('--compare', action='store_true', help='Run comparison examples')
    parser.add_argument('-all', action='store_true', help='Calculate all metrics including WER, CER, and others')
    parser.add_argument('--csv', action='store_true', help='Export results to CSV file')
    parser.add_argument('--ipa', action='store_true', help='Use direct IPA conversion with phonemizer instead of G2P')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Define output paths
    weights_path = os.path.join(args.save_dir, 'viseme_weights.json') if not args.weights else args.weights
    if args.ipa:
        # Use separate weights file for IPA-based calculations if no explicit path provided
        if not args.weights:
            weights_path = os.path.join(args.save_dir, 'viseme_weights_ipa.json')
    results_path = os.path.join(args.save_dir, 'results.json')
    metrics_path = os.path.join(args.save_dir, 'metrics.txt')
    
    # Create weighted evaluator with IPA option if specified
    evaluator = WeightedLipReadingEvaluator(use_ipa=args.ipa)
    
    # Print phoneme conversion approach
    if args.ipa:
        print("\n=== USING DIRECT IPA CONVERSION ===")
        print("Results will be saved with '_ipa' suffix.")
    else:
        print("\n=== USING G2P (GRAPHEME-TO-PHONEME) CONVERSION ===")
    
    print("Note: Weights are now always calculated from scratch")
    
    # Quick conversion test with a sentence containing 'oy' diphthong
    test_sentence = "I enjoy toys and boys playing outside."
    phonemes = evaluator.text_to_phonemes(test_sentence)
    print(f"Test conversion: '{test_sentence}'")
    print(f"Phonemes: {' '.join(phonemes)}")
    visemes = [evaluator.phoneme_to_viseme.get(p, -1) for p in phonemes]
    print(f"Visemes: {' '.join(str(v) for v in visemes)}")
    print()
    
    # Save weights
    if evaluator.feature_weights:
        evaluator.save_weights_to_file(weights_path)
        print(f"Saved computed weights to {weights_path}")
    
    # Run comparison examples if requested
    if args.compare:
        example_pairs = [
            ("Hello world", "Hello word"),   # Small difference
            ("Please pass the salt", "Please pass the fault"),   # Visually similar sounds
            ("Bring me a glass of water", "Bring me a glass of vodka"),  # Visually similar 'w' vs 'v'
            ("Meet me at five", "Meet me at nine"),  # 'f' vs 'n' (different visemes)
            ("I enjoy boys playing outside", "I enjoy toys playing outside"),  # 'b' vs 't' with 'oy' diphthong
        ]
        
        print("\n=== Comparing Standard vs. Weighted Viseme Alignment ===\n")
        
        for ref, hyp in example_pairs:
            print(f"\nReference: '{ref}'")
            print(f"Hypothesis: '{hyp}'")
            
            # Evaluate with both approaches
            results = evaluator.evaluate_pair_with_details(ref, hyp)
            
            # Print results
            print(f"Standard viseme score: {results['standard']['viseme_alignment_score']:.3f}")
            print(f"Weighted viseme score: {results['weighted']['viseme_alignment_score']:.3f}")
            viseme_diff = results['weighted']['viseme_alignment_score'] - results['standard']['viseme_alignment_score']
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
            print(f"\nAnalysis complete! Using {'IPA' if args.ipa else 'G2P'} conversion.")
            print(f"All outputs saved to: {args.save_dir}")


if __name__ == "__main__":
    main() 