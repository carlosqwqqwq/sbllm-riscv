# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .parser.DFG import DFG_cpp
from .adapter import get_parser

def calc_data_flow_match(references, candidate, lang, tree_sitter_lang_obj):
    parser = get_parser(tree_sitter_lang_obj)
    
    def get_data_flow(code):
        tree = parser.parse(bytes(code, 'utf8'))
        root_node = tree.root_node
        # For simplicity in this integration, we use the DFG extraction
        # to get a set of data flow relationships
        return set(DFG_cpp(root_node, code.split('\n')))

    candidate_dfg = get_data_flow(candidate)
    
    total_score = 0
    for reference in references:
        reference_dfg = get_data_flow(reference)
        
        if not reference_dfg:
            # If no data flow in reference, semantic match is 1.0 if candidate also has none
            score = 1.0 if not candidate_dfg else 0.0
        else:
            match_count = len(candidate_dfg.intersection(reference_dfg))
            score = match_count / len(reference_dfg)
            
        total_score = max(total_score, score)
        
    return total_score
