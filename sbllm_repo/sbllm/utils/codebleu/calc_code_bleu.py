# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from .weighted_ngram_match import calc_weighted_ngram_match
from .syntax_match import calc_syntax_match
from .dataflow_match import calc_data_flow_match
from ..code_utils import remove_comments_and_docstrings
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def compute_codebleu(references, candidate, lang, tree_sitter_lang_obj, keywords=[], weights=(0.25, 0.25, 0.25, 0.25)):
    """
    Main entry point for CodeBLEU calculation.
    """
    # 1. N-gram match (BLEU)
    # references is a list of lists of tokens, candidate is a list of tokens
    chencherry = SmoothingFunction()
    ref_tokens = [[r.split()] for r in references]
    cand_tokens = [candidate.split()]
    ngram_score = corpus_bleu(ref_tokens, cand_tokens, smoothing_function=chencherry.method1)
    
    # 2. Weighted N-gram match
    dict_weights = {kw: 5 for kw in keywords}
    weighted_ngram_score = calc_weighted_ngram_match(references, candidate, [0.25, 0.25, 0.25, 0.25], dict_weights)
    
    # 3. Syntax match
    syntax_score = calc_syntax_match(references, candidate, lang, tree_sitter_lang_obj)
    
    # 4. Dataflow match
    dataflow_score = calc_data_flow_match(references, candidate, lang, tree_sitter_lang_obj)
    
    # Combine scores
    total_score = (weights[0] * ngram_score + 
                   weights[1] * weighted_ngram_score + 
                   weights[2] * syntax_score + 
                   weights[3] * dataflow_score)
                   
    return {
        'codebleu': total_score,
        'ngram': ngram_score,
        'weighted_ngram': weighted_ngram_score,
        'syntax': syntax_score,
        'dataflow': dataflow_score
    }
