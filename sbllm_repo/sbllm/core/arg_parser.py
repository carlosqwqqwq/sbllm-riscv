import os
import argparse
import logging
import random
import numpy as np


logger = logging.getLogger(__name__)



def cfg_parsing():
    parser = argparse.ArgumentParser()
    # Public parameters
    parser.add_argument("--mode", default=None, type=str, required=True)   
    parser.add_argument("--lang", default=None, type=str, required=True)   
    parser.add_argument("--dataset_type", default="standard", type=str, 
                        help="Type of dataset: standard, project_rvv, project_vecintrin")
    parser.add_argument("--output_path", default=None, type=str,
                        help="The path for output.")
    parser.add_argument("--seed", default=42, type=int,
                        help="The random seed.")
    
    # DPP parameters
    parser.add_argument("--num_candidates", default=16, type=int, 
                        help="The number of groups for training.")
    parser.add_argument("--num_icl", default=4, type=int, 
                        help="The number of examples in ICL.")  
    parser.add_argument("--num_process", default=8, type=int, 
                        help="The number of multi process.")  
    parser.add_argument("--dpp_topk", default=100, type=int,
                        help="The size of DPP matrix.")
    parser.add_argument("--scale_factor", default=0.1, type=float,
                        help="The factor to trade-off diversity and relevance.")
    parser.add_argument("--dimension", default=768, type=int,
                        help="The dimension of embedding.")
    
    # Inference parameters
    parser.add_argument("--model_name", default=None, type=str, required=True)
    parser.add_argument("--baseline_data_path", default='', type=str)
    parser.add_argument("--training_data_path", default='', type=str)
    parser.add_argument("--generation_path", default=None, type=str)
    parser.add_argument("--master_port", default='1', type=str)
    parser.add_argument("--generation_model_path", default=None, type=str) 
    parser.add_argument("--instruction", default='# optimize this code \n# slow version\n', type=str)
    parser.add_argument("--middle_instruction", default='\n# optimized version of the same code\n', type=str)
    parser.add_argument("--post_instruction", default='\n\n\n', type=str)
    parser.add_argument("--temperature", default=0, type=float)  
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--slice", default=1, type=int)
    parser.add_argument("--total", default=1, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--max_gen_length", default=256, type=int)
    parser.add_argument("--max_seq_length", default=4096, type=int)
    parser.add_argument("--test_batch_size", default=32, type=int)
    parser.add_argument("--testing_number", default=0, type=int)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--iteration", default=1, type=int)
    parser.add_argument("--beam_number", default=1, type=int)
    parser.add_argument("--generation_number", default=1, type=int)
    parser.add_argument("--api_idx", default=0, type=int)
    parser.add_argument("--restart_pos", default=0, type=int)
    parser.add_argument("--from_file", action='store_true')
    

    # Training parameters
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--num_train_epochs", default=10, type=int)
    parser.add_argument("--test_case_path", default=None, type=str)
    parser.add_argument("--process_number", default=30, type=int)
    parser.add_argument("--train_batch_size", default=1, type=int)
    parser.add_argument("--valid_batch_size", default=32, type=int)
    parser.add_argument("--margin", default=0.3, type=float)

    # Retrieval parameters
    parser.add_argument("--retrieval_method", default="bm25", type=str,
                        choices=["bm25", "vector", "hybrid"],
                        help="Retrieval method: bm25, vector, or hybrid")
    parser.add_argument("--hybrid_alpha", default=0.5, type=float,
                        help="Weight for BM25 score in hybrid retrieval (0-1)")
    parser.add_argument("--vector_index_path", default=None, type=str,
                        help="Path to FAISS vector index file")
    parser.add_argument("--embedding_model", default="all-MiniLM-L6-v2", type=str,
                        help="Sentence transformer model name for embeddings")

    # RISC-V specific parameters
    parser.add_argument("--qemu_path", default="/usr/local/bin/qemu-riscv64", type=str,
                        help="Path to QEMU executable (in-container or local)")
    parser.add_argument("--riscv_gcc_toolchain_path", default="/opt/riscv", type=str,
                        help="Path to RISC-V GCC toolchain directory (in-container or local)")
    parser.add_argument("--riscv_mode", action='store_true',
                        help="Enable RISC-V optimization mode")
    parser.add_argument("--use_docker", action='store_true',
                        help="Use Docker (riscv-opt-env) for evaluation instead of local environment")

    # Hyperparameters for Evaluation and Scoring
    parser.add_argument("--eval_num_runs", default=5, type=int,
                        help="Number of times to run each program for stable timing.")
    parser.add_argument("--eval_timeout", default=60, type=int,
                        help="Timeout in seconds for execution (per run).")
    parser.add_argument("--compile_timeout", default=30, type=int,
                        help="Timeout in seconds for compilation.")
    parser.add_argument("--score_alpha", default=0.7, type=float,
                        help="Weight for speed improvement in composite score (0-1).")
    parser.add_argument("--score_beta", default=0.3, type=float,
                        help="Weight for size reduction in composite score (0-1).")
    parser.add_argument("--workspace_timeout", default=10, type=int,
                        help="Timeout for acquiring an isolated workspace.")

    cfg = parser.parse_args()
    
    # --- Auto-inference and Validation ---
    if not cfg.output_path and cfg.generation_path:
        cfg.output_path = cfg.generation_path
        logger.info("Syncing output_path with generation_path")

    if not cfg.generation_path and cfg.output_path:
        cfg.generation_path = cfg.output_path
        logger.info("Syncing generation_path with output_path")

    if cfg.qemu_path and not cfg.riscv_mode:
        cfg.riscv_mode = True
        logger.info("Auto-enabled riscv_mode based on qemu_path")

    if cfg.output_path:
        os.makedirs(cfg.output_path, exist_ok=True)
    
    if cfg.lang == 'riscv' and not cfg.riscv_mode:
        cfg.riscv_mode = True

    # Seed initialization
    random.seed(cfg.seed)
    os.environ['PYHTONHASHSEED'] = str(cfg.seed)
    np.random.seed(cfg.seed)
    
    return cfg

