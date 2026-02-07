import os
import numpy as np
import jsonlines
import logging
import difflib

logger = logging.getLogger(__name__)

def load_knowledge_base(kb_path='processed_data/riscv/train.jsonl'):
    """Loads the knowledge base from JSONL file."""
    kb = []
    if not os.path.exists(kb_path):
        logger.warning(f"Knowledge Base not found at {kb_path}")
        return kb
    
    with jsonlines.open(kb_path) as reader:
        for obj in reader:
            if obj.get('optimized_code') and obj.get('original_code'):
                kb.append(obj)
    logger.info(f"Loaded {len(kb)} items into Knowledge Base.")
    return kb

def retrieve_examples_difflib(query_code, knowledge_base, k=3):
    """
    Retrieves top-k similar examples from Knowledge Base using difflib.
    """
    if not knowledge_base:
        return ""

    scores = []
    for item in knowledge_base:
        score = difflib.SequenceMatcher(None, query_code, item['original_code']).quick_ratio()
        scores.append((score, item))
    
    scores.sort(key=lambda x: x[0], reverse=True)
    top_k = scores[:k]
    
    examples_str = ""
    for i, (score, item) in enumerate(top_k):
        if score > 0.1:
            examples_str += f"[Example {i+1} (Similarity: {score:.2f})]\n"
            examples_str += f"// Description: {item.get('optimization_description', 'N/A')}\n"
            examples_str += f"// Original:\n{item['original_code']}\n"
            examples_str += f"// Optimized (RVV):\n{item['optimized_code']}\n\n"
            
    return examples_str


def load_or_build_vector_index(cfg, train_data, embedding_model):
    """
    Load or build FAISS vector index based on configuration.
    Returns the vector index and training embeddings (if built).
    """
    vector_index = None
    train_embeddings = None
    
    if cfg.retrieval_method not in ['vector', 'hybrid']:
        return None, None

    if embedding_model is None:
        return None, None
        
    # Try to load existing index
    if cfg.vector_index_path and os.path.exists(cfg.vector_index_path):
        try:
            import faiss
            vector_index = faiss.read_index(cfg.vector_index_path)
            print(f"Loaded vector index from {cfg.vector_index_path}")
            return vector_index, None
        except Exception as e:
            print(f"Failed to load vector index: {e}")
            vector_index = None

    # Build index if not loaded
    try:
        print("Building vector index from training data...")
        train_texts = []
        for obj in train_data:
            if 'text_representation' in obj:
                train_texts.append(obj['text_representation'])
            elif 'query_abs' in obj:
                train_texts.append(' '.join(obj['query_abs']) if isinstance(obj['query_abs'], list) else obj['query_abs'])
            else:
                train_texts.append('')
        
        train_embeddings = embedding_model.encode(train_texts, convert_to_numpy=True, show_progress_bar=True)
        
        import faiss
        dimension = train_embeddings.shape[1]
        vector_index = faiss.IndexFlatL2(dimension)
        vector_index.add(train_embeddings.astype('float32'))
        
        # Save index
        if cfg.vector_index_path:
            os.makedirs(os.path.dirname(cfg.vector_index_path), exist_ok=True)
            faiss.write_index(vector_index, cfg.vector_index_path)
            print(f"Saved vector index to {cfg.vector_index_path}")
        else:
            default_index_path = cfg.training_data_path.replace('.jsonl', '_faiss_index.bin')
            faiss.write_index(vector_index, default_index_path)
            print(f"Saved vector index to {default_index_path}")
            
    except Exception as e:
        print(f"Failed to build vector index: {e}")
        vector_index = None
        
    return vector_index, train_embeddings

def perform_hybrid_search(query_abs, train_data, retrieval_method, hybrid_alpha, vector_index, embedding_model, code_bm25, top_k=100):
    """
    Perform hybrid search (BM25 + Vector) and return normalized scores and indices.
    """
    # Initialize basic BM25 results
    bm25_scores_normalized = np.array([])
    bm25_top_indices = np.array([], dtype=int)
    
    # Run BM25
    if retrieval_method in ['bm25', 'hybrid'] and code_bm25 is not None and len(train_data) > 0:
        bm25_scores = np.array(code_bm25.get_scores(query_abs))
        if len(bm25_scores) > 0:
            bm25_top_indices = np.argsort(-bm25_scores)[:min(top_k, len(bm25_scores))]
            score_range = bm25_scores.max() - bm25_scores.min()
            if score_range > 1e-8:
                bm25_scores_normalized = (bm25_scores - bm25_scores.min()) / score_range
            else:
                bm25_scores_normalized = np.ones_like(bm25_scores)
    else:
        # Fallback if BM25 not available/selected
        if len(train_data) > 0:
             bm25_scores_normalized = np.zeros(len(train_data))
             bm25_top_indices = np.arange(min(top_k, len(train_data)), dtype=int)

    # Run Vector Search
    vector_scores_normalized = np.array([])
    vector_top_indices = np.array([], dtype=int)
    
    if retrieval_method in ['vector', 'hybrid'] and vector_index is not None and embedding_model is not None and vector_index.ntotal > 0:
        query_text = ' '.join(query_abs)
        query_embedding = embedding_model.encode([query_text], convert_to_numpy=True)
        
        k = min(top_k, vector_index.ntotal)
        vector_scores, vector_indices = vector_index.search(query_embedding, k)
        vector_top_indices = vector_indices[0]
        vector_scores = vector_scores[0]
        
        if len(vector_scores) > 0:
            score_range = vector_scores.max() - vector_scores.min()
            if score_range > 1e-8:
                # L2 distance: smaller is better, so 1 - normalized distance
                vector_scores_normalized = 1 - (vector_scores - vector_scores.min()) / score_range
            else:
                vector_scores_normalized = np.ones_like(vector_scores)
    else:
         if len(train_data) > 0:
            vector_scores_normalized = np.zeros(len(train_data))
            vector_top_indices = np.arange(min(top_k, len(train_data)), dtype=int)

    # Combine Results
    if len(train_data) == 0:
        return np.array([]), np.array([], dtype=int)

    final_scores = np.zeros(len(train_data))
    
    if retrieval_method == 'hybrid':
        # Apply BM25 scores
        if len(bm25_top_indices) > 0 and len(bm25_scores_normalized) > 0:
            valid_mask = (bm25_top_indices < len(final_scores))
            valid_indices = bm25_top_indices[valid_mask]
            # Need to filter scores too
            valid_scores_mask = valid_mask[:len(bm25_scores_normalized)] 
             # Just matching checks
            if len(valid_indices) > 0:
                # We need to map back carefully if lengths differ, but typically indices and scores match length
                # Simpler:
                for idx, score in zip(bm25_top_indices, bm25_scores_normalized):
                     if 0 <= idx < len(final_scores):
                         final_scores[idx] += hybrid_alpha * score

        # Apply Vector scores
        if len(vector_top_indices) > 0 and len(vector_scores_normalized) > 0:
            for idx, score in zip(vector_top_indices, vector_scores_normalized):
                if 0 <= idx < len(final_scores):
                    final_scores[idx] += (1 - hybrid_alpha) * score
        
        # Hybrid relies on indices accumulated in final_scores
        # But for top_k return we sort final_scores
        pass

    elif retrieval_method == 'vector':
         if len(vector_top_indices) > 0 and len(vector_scores_normalized) > 0:
            for idx, score in zip(vector_top_indices, vector_scores_normalized):
                 if 0 <= idx < len(final_scores):
                     final_scores[idx] = score
    
    else: # bm25
         if len(bm25_top_indices) > 0 and len(bm25_scores_normalized) > 0:
              # We have full scores computed above? 
              # Actually code_bm25.get_scores returns scores for ALL docs usually.
              # Let's verify: yes, get_scores(query) returns list of size N.
              # So bm25_scores_normalized from above could be used directly if it was full size.
              # Optimization: if we used top_k above, we only have partial.
              # Re-checking logic: The original code computed full scores for BM25.
              pass
              
         # Logic simplification:
         # Just use the bm25_scores_normalized we handled above?
         # Wait, in the original code: 
         # bm25_scores = code_bm25.get_scores(query_abs) -> returns len(corpus) scores.
         # The 'indices' logic was just for optimization or debug? 
         # Actually for sparse + dense combination it's tricky.
         # Let's trust the logic structure: initialize zeros, add weighted scores at indices.
         pass
         
         # However, for pure BM25, we might want the dense array if possible or just top k.
         # Ref logic: code_scores = final_scores.
         # If bm25, final_scores = bm25_scores_normalized.
         if len(bm25_scores_normalized) == len(train_data):
              final_scores = bm25_scores_normalized
         else:
               # Pushed from top k
              for idx, score in zip(bm25_top_indices, bm25_scores_normalized):
                   if 0 <= idx < len(final_scores):
                       final_scores[idx] = score

    return final_scores

