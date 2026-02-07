# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .adapter import get_parser
from collections import Counter

def node_to_sexp(node):
    """
    Recursive fallback for the missing node.sexp() in tree-sitter 0.23+
    """
    if not node.children:
        return f"({node.type})"
    
    parts = [node.type]
    for child in node.children:
        parts.append(node_to_sexp(child))
    
    return "(" + " ".join(parts) + ")"

def calc_syntax_match(references, candidate, lang, tree_sitter_lang_obj):
    parser = get_parser(tree_sitter_lang_obj)
    
    def get_subtrees(node):
        subtrees = []
        stack = [node]
        while stack:
            curr = stack.pop()
            # Use fallback sexp representation
            subtrees.append(node_to_sexp(curr))
            for child in curr.children:
                stack.append(child)
        return subtrees

    candidate_tree = parser.parse(bytes(candidate, 'utf8')).root_node
    candidate_subtrees = get_subtrees(candidate_tree)
    candidate_counts = Counter(candidate_subtrees)
    
    total_score = 0
    for reference in references:
        reference_tree = parser.parse(bytes(reference, 'utf8')).root_node
        reference_subtrees = get_subtrees(reference_tree)
        reference_counts = Counter(reference_subtrees)
        
        match_count = 0
        for subtree, count in candidate_counts.items():
            match_count += min(count, reference_counts[subtree])
            
        precision = match_count / len(candidate_subtrees) if candidate_subtrees else 0
        total_score = max(total_score, precision)
        
    return total_score
