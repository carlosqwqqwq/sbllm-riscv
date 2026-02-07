import logging
import os
import tree_sitter
from tree_sitter import Language

logger = logging.getLogger(__name__)

# RISC-V RVV intrinsics keywords for weighting
RISCV_KEYWORDS = [
    # Vector types
    "vint8m1_t", "vint8m2_t", "vint8m4_t", "vint8m8_t",
    "vuint8m1_t", "vuint8m2_t", "vuint8m4_t", "vuint8m8_t",
    "vint16m1_t", "vint16m2_t", "vint16m4_t", "vint16m8_t",
    "vint32m1_t", "vint32m2_t", "vint32m4_t", "vint32m8_t",
    "vuint32m1_t", "vuint32m2_t", "vuint32m4_t", "vuint32m8_t",
    "vfloat32m1_t", "vfloat32m2_t", "vfloat32m4_t", "vfloat32m8_t",
    "vbool1_t", "vbool2_t", "vbool4_t", "vbool8_t", "vbool16_t",
    # Vector load/store
    "vle8_v", "vle16_v", "vle32_v", "vle64_v",
    "vse8_v", "vse16_v", "vse32_v", "vse64_v",
    "vluxei32_v", "vsuxei32_v",
    # Vector arithmetic
    "vadd_vv", "vadd_vx", "vsub_vv", "vsub_vx",
    "vmul_vv", "vmul_vx", "vdiv_vv", "vdiv_vx",
    "vmacc_vv", "vnmsac_vv", "vmadd_vv",
    # Vector bitwise/compare
    "vand_vv", "vor_vv", "vxor_vv", "vnot_v",
    "vmseq_vv", "vmsne_vv", "vmslt_vv",
    # Vector configuration/permutation
    "vsetvl_e8m1", "vsetvl_e16m1", "vsetvl_e32m1", "vsetvl_e64m1",
    "__riscv_vsetvl_e32m1", "vslidedown_vx", "vslideup_vx", "vrgather_vv"
]

_CPP_LANG = None
_CODEBLEU_AVAILABLE = False

try:
    from .codebleu.calc_code_bleu import compute_codebleu
    import tree_sitter_cpp
    _CODEBLEU_AVAILABLE = True
except ImportError:
    # Quietly fail on host; will fallback to Docker if needed
    _CODEBLEU_AVAILABLE = False

from .codebleu.adapter import get_language

def get_tree_sitter_lang(lang):
    global _CPP_LANG
    if not _CODEBLEU_AVAILABLE:
        return None
        
    if lang in ['c', 'cpp', 'riscv']:
        if _CPP_LANG is None:
            try:
                # Use robust adapter
                _CPP_LANG = get_language(tree_sitter_cpp.language(), "cpp")
            except Exception as e:
                logger.error(f"Failed to load tree-sitter-cpp: {e}")
                return None
        return _CPP_LANG
    return None

def get_codebleu_score(reference: str, hypothesis: str, lang: str = 'riscv'):
    """
    Calculate CodeBLEU score using official logic.
    """
    if not _CODEBLEU_AVAILABLE:
        return 0.0

    if not reference or not hypothesis:
        return 0.0
        
    lang_obj = get_tree_sitter_lang(lang)
    if lang_obj is None:
        return 0.0
        
    try:
        results = compute_codebleu(
            [reference], 
            hypothesis, 
            'cpp', 
            lang_obj, 
            keywords=RISCV_KEYWORDS if lang == 'riscv' else []
        )
        return results['codebleu']
    except Exception as e:
        logger.error(f"Error in CodeBLEU calculation: {e}")
        return 0.0

def get_detailed_codebleu(reference: str, hypothesis: str, lang: str = 'riscv'):
    """
    Get detailed CodeBLEU metrics.
    """
    if not _CODEBLEU_AVAILABLE:
        return None

    if not reference or not hypothesis:
        return None
        
    lang_obj = get_tree_sitter_lang(lang)
    if lang_obj is None:
        return None
        
    try:
        return compute_codebleu(
            [reference], 
            hypothesis, 
            'cpp', 
            lang_obj, 
            keywords=RISCV_KEYWORDS if lang == 'riscv' else []
        )
    except Exception as e:
        logger.error(f"Error in detailed CodeBLEU calculation: {e}")
        return None

def batch_get_codebleu_docker(items, lang='riscv', project_root=None):
    """
    Calculate CodeBLEU for multiple items inside a Docker container.
    'items' is a list of (reference, hypothesis) tuples.
    """
    import subprocess
    import json
    import tempfile
    
    if not items:
        return []
        
    if project_root is None:
        project_root = os.getcwd()

    # Use a temp file inside the project root for container access
    # Results directory is a good place
    temp_dir = os.path.join(project_root, 'results', 'temp_codebleu')
    os.makedirs(temp_dir, exist_ok=True)
    
    input_file = os.path.join(temp_dir, 'input.json')
    output_file = os.path.join(temp_dir, 'output.json')
    
    try:
        with open(input_file, 'w', encoding='utf-8') as f:
            json.dump(items, f)
            
        # Docker Command
        # This calls back into this same module inside the container
        cmd = [
            'docker', 'run', '--rm',
            '-v', f"{os.path.abspath(project_root)}:/work",
            '-w', '/work',
            'riscv-opt-env',
            'python3', '-c',
            f"import sys, json, os; sys.path.append('/work'); " +
            f"from sbllm.utils.codebleu_utils import get_detailed_codebleu; " +
            f"data = json.load(open('/work/results/temp_codebleu/input.json', 'r')); " +
            f"results = [get_detailed_codebleu(r[0], r[1], lang='{lang}') for r in data]; " +
            f"json.dump(results, open('/work/results/temp_codebleu/output.json', 'w'))"
        ]
        
        # logger.info(f"Running batch CodeBLEU in Docker for {len(items)} items...")
        cp = subprocess.run(cmd, capture_output=True, text=True)
        if cp.returncode != 0:
            logger.error(f"Docker batch CodeBLEU failed with exit code {cp.returncode}")
            logger.error(f"STDOUT: {cp.stdout}")
            logger.error(f"STDERR: {cp.stderr}")
            return [None] * len(items)
        
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Docker batch CodeBLEU failed: {e}")
    finally:
        # Cleanup
        try:
            if os.path.exists(input_file): os.remove(input_file)
            if os.path.exists(output_file): os.remove(output_file)
        except: pass
        
    return [None] * len(items)
