# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
from collections import Counter

def calc_weighted_ngram_match(references, candidate, weights, dict_weights):
    def get_weighted_ngrams(tokens, n, dict_weights):
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            # Weight is the average weight of tokens in the ngram
            weight = sum(dict_weights.get(t, 1) for t in ngram) / n
            ngrams.append((ngram, weight))
        return ngrams

    candidate_tokens = candidate.split()
    max_n = len(weights)
    
    scores = []
    for n in range(1, max_n + 1):
        candidate_weighted_ngrams = get_weighted_ngrams(candidate_tokens, n, dict_weights)
        if not candidate_weighted_ngrams:
            scores.append(0)
            continue
            
        candidate_counts = Counter(ngram for ngram, weight in candidate_weighted_ngrams)
        candidate_weights = {ngram: weight for ngram, weight in candidate_weighted_ngrams}
        
        max_counts = Counter()
        for reference in references:
            reference_tokens = reference.split()
            reference_ngrams = [tuple(reference_tokens[i:i+n]) for i in range(len(reference_tokens) - n + 1)]
            reference_counts = Counter(reference_ngrams)
            for ngram in candidate_counts:
                max_counts[ngram] = max(max_counts[ngram], reference_counts[ngram])
        
        match_count = 0
        total_weighted_count = sum(candidate_weights[ngram] * count for ngram, count in candidate_counts.items())
        
        for ngram, count in candidate_counts.items():
            match_count += candidate_weights[ngram] * min(count, max_counts[ngram])
            
        if total_weighted_count == 0:
            scores.append(0)
        else:
            scores.append(match_count / total_weighted_count)
            
    # Calculate geometric mean
    if all(s > 0 for s in scores):
        score = math.exp(sum(w * math.log(s) for w, s in zip(weights, scores) if s > 0))
    else:
        score = 0
        
    return score
