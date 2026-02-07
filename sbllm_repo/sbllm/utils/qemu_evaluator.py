"""
QEMU RISC-V 评估器模块

用于在 QEMU 仿真环境中运行和评估 RISC-V 代码的性能。
支持功能正确性检查、性能评估和代码大小分析。
"""
import os
import re
import hashlib
import shutil
import subprocess
import tempfile
import time
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from sbllm.core.evaluator_interface import BaseEvaluator, EvalMetric
from .riscv_compiler import RISCVCompiler

logger = logging.getLogger(__name__)

class QEMURISCVEvaluator(BaseEvaluator):
    """
    QEMU RISC-V 评估器类
    """
    
    def __init__(self, qemu_path: str, riscv_gcc_toolchain_path: str, temp_dir: Optional[str] = None,
                 enable_cache: bool = True, max_workers: int = 4, 
                 use_docker: bool = False, project_root: Optional[str] = None):
        self.qemu_path = qemu_path
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.max_workers = max_workers
        self.riscv_gcc_toolchain_path = riscv_gcc_toolchain_path
        self.use_docker = use_docker
        self.project_root = project_root or os.getcwd()
        
        # Binary utilities
        # Support riscv64-unknown-linux-gnu-size (default in riscv-opt-env)
        size_name = 'riscv64-unknown-linux-gnu-size'
        
        # Check for bin directory, or just use root
        bin_dir = os.path.join(riscv_gcc_toolchain_path, 'bin')
        # On Windows, if we are using Docker, the container path /opt/riscv/bin won't exist locally.
        # We should skip the existence check if use_docker is true.
        if not use_docker and not os.path.exists(bin_dir):
            bin_dir = riscv_gcc_toolchain_path
            
        self.riscv_size_path = os.path.join(bin_dir, size_name).replace('\\', '/')
        
        # Determine if we should skip existence check (Docker or WSL Mode on Windows)
        skip_check = (os.name == 'nt' and (use_docker or riscv_gcc_toolchain_path.startswith('/') or riscv_gcc_toolchain_path.startswith('/mnt/')))
        
        if not skip_check and not os.path.exists(self.riscv_size_path):
             # Fallback check for linux-gnu (stay with local paths for existence check)
             fallback = os.path.join(bin_dir, 'riscv64-linux-gnu-size')
             if os.path.exists(fallback):
                 self.riscv_size_path = fallback.replace('\\', '/')
             else:
                 raise FileNotFoundError(f"RISC-V size path not found: {self.riscv_size_path}")

        # Normalize paths for Linux container
        self.qemu_path = qemu_path.replace('\\', '/')

        self.compiler = RISCVCompiler(riscv_gcc_toolchain_path, self.temp_dir, enable_cache, use_docker, self.project_root)
        
        if not skip_check and not os.path.exists(self.qemu_path):
            raise FileNotFoundError(f"QEMU path not found: {self.qemu_path}")

    def build_env(self):
        """Setup build environment (No-op as checks are in __init__)."""
        pass

    def to_container_path(self, p: str) -> str:
        """Translates a host path to a container path using volume mount mapping."""
        if not self.use_docker:
            if os.name == 'nt' and ':' in p:
                drive, rest = p.split(':', 1)
                linux_rest = rest.replace('\\', '/')
                return f"/mnt/{drive.lower()}{linux_rest}"
            return p.replace('\\', '/')

        abs_p = os.path.abspath(p)
        abs_root = os.path.abspath(self.project_root)
        if abs_p.lower().startswith(abs_root.lower()):
            rel_path = os.path.relpath(abs_p, abs_root)
            return os.path.join('/work', rel_path).replace('\\', '/')
        return abs_p.replace('\\', '/')

    def evaluate(self, code: str, metadata: Dict[str, Any], **kwargs) -> EvalMetric:
        # standard extraction of hyperparameters
        num_runs = kwargs.get('num_runs', 5)
        eval_timeout = kwargs.get('timeout', 60)
        compile_timeout = kwargs.get('compile_timeout', 30)
        
        input_data = metadata.get("input_data", "")
        reference_output = metadata.get("reference_output", None)
        
        with tempfile.TemporaryDirectory(dir=self.temp_dir) as workspace:
            candidate_bin = os.path.join(workspace, "candidate.bin")

            if not self.compiler.compile(code, candidate_bin, timeout=compile_timeout):
                return EvalMetric(compile_success=False, error="Candidate compilation failed")

            exec_time, code_size, is_correct, output = self.run_and_measure(
                candidate_bin, input_data, num_runs=num_runs, 
                reference_output=reference_output, timeout=eval_timeout
            )
            
            return EvalMetric(
                time=exec_time,
                size=code_size,
                correct=is_correct,
                output=output,
                compile_success=True
            )

    def run_and_measure(self, bin_path: str, input_data: str = "", num_runs: int = 5, 
                        reference_output: Optional[str] = None, timeout: int = 60) -> Tuple[float, int, bool, str]:
        if not os.path.exists(bin_path):
            return 99999.0, 0, False, ""
        
        code_size = self._get_code_size(bin_path)
        execution_times = []
        outputs = []
        all_correct = True
        
        # Base QEMU command (container-native paths)
        qemu_exec_cmd = [self.qemu_path, self.to_container_path(bin_path)]
        
        # Wrap with Docker, WSL or native
        if self.use_docker:
            from sbllm.utils.docker_manager import docker_manager
            docker_manager.initialize(self.project_root)
            
            for _ in range(num_runs):
                try:
                    start_time = time.perf_counter()
                    result = docker_manager.exec(qemu_exec_cmd, input_data=input_data, timeout=timeout)
                    end_time = time.perf_counter()
                    
                    execution_times.append(end_time - start_time)
                    output = result.stdout.strip()
                    outputs.append(output)
                    
                    if result.returncode != 0:
                        all_correct = False
                        logger.error(f"Execution failed (Docker-Exec, Code={result.returncode}):\n{result.stderr}")
                    
                    if reference_output is not None:
                        if not self._compare_outputs(output, reference_output):
                            all_correct = False
                        
                except Exception as e:
                    logger.error(f"Error during execution (Docker-Exec): {e}")
                    execution_times.append(float('inf'))
                    all_correct = False
        else:
            # WSL or Native fallback
            exec_args = []
            if os.name == 'nt':
                exec_args = ['wsl'] + qemu_exec_cmd
            else:
                exec_args = qemu_exec_cmd

            for _ in range(num_runs):
                try:
                    start_time = time.perf_counter()
                    result = subprocess.run(exec_args, input=input_data, capture_output=True, text=True, timeout=timeout)
                    end_time = time.perf_counter()
                    
                    execution_times.append(end_time - start_time)
                    output = result.stdout.strip()
                    outputs.append(output)
                    
                    if result.returncode != 0:
                        all_correct = False
                    
                    if reference_output is not None:
                        if not self._compare_outputs(output, reference_output):
                            all_correct = False
                except Exception:
                    execution_times.append(float('inf'))
                    all_correct = False
        
        finite_times = [t for t in execution_times if t != float('inf')]
        avg_time = sum(finite_times) / len(finite_times) if finite_times else 99999.0
        if not execution_times or float('inf') in execution_times: all_correct = False
        final_output = max(set(outputs), key=outputs.count) if outputs else ""
        
        return avg_time, code_size, all_correct, final_output

    def _get_code_size(self, binary_path: str) -> int:
        try:
            size_bin = self.to_container_path(binary_path)
            # Ensure command path uses forward slashes
            size_cmd = [self.riscv_size_path.replace('\\', '/'), size_bin]
            
            if self.use_docker:
                from sbllm.utils.docker_manager import docker_manager
                docker_manager.initialize(self.project_root)
                result = docker_manager.exec(size_cmd, timeout=10)
            else:
                full_cmd = []
                if os.name == 'nt':
                    full_cmd = ['wsl'] + size_cmd
                else:
                    full_cmd = size_cmd
                result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    parts = lines[1].split()
                    if len(parts) >= 1: return int(parts[0])
            return 0
        except Exception as e:
            logger.warning(f"Failed to get code size: {e}")
            return 0

    def _compare_outputs(self, output1: str, output2: str, tolerance: float = 1e-4) -> bool:
        """
        Compare two outputs. 
        Protocol Update (Round 19.5): 
        If output contains '[VERIFY]', only compare lines starting with '[VERIFY]'.
        This allows ignoring '[TIME]' or other debug info in the output.
        """
        def filter_verify(text):
            lines = text.strip().splitlines()
            v_lines = [l for l in lines if l.startswith('[VERIFY]')]
            # If no verify tags found, fallback to all lines (backward compatibility)
            return v_lines if v_lines else lines

        lines1 = filter_verify(output1)
        lines2 = filter_verify(output2)
        
        if len(lines1) != len(lines2): 
            # logger.debug(f"Line count mismatch: {len(lines1)} vs {len(lines2)}")
            return False
        
        for l1, l2 in zip(lines1, lines2):
            l1, l2 = l1.strip(), l2.strip()
            if l1 == l2: continue
            
            # Try float comparison for roughly matching numbers
            # Extract numbers if the line follows "Label: value" format
            try:
                # Simple heuristic: try to compare the last token as float
                val1 = float(l1.split()[-1])
                val2 = float(l2.split()[-1])
                if abs(val1 - val2) > tolerance: 
                    return False
            except (ValueError, IndexError):
                # If structure differs or not float, strictly fail
                return False
        return True

class StandaloneEvaluator(QEMURISCVEvaluator):
    """
    Evaluator for Single-File Standalone Benchmarks.
    Wraps candidate code with a C++ harness and compiles/runs as a single executable.
    Uses Header/Footer structure with dynamic function name extraction (Round 16).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Dynamic import to get benchmark configurations
        from . import harness_standalone
        self.harness_module = harness_standalone

    def _extract_function_name(self, code: str, benchmark_name: str) -> str:
        """
        Extract the actual function name from LLM-generated code.
        Looks for function DEFINITIONS (with body), not declarations.
        """
        config = self.harness_module.get_benchmark_config(benchmark_name)
        if not config:
            return None
        
        default_func = config.get('default_func')
        
        # Strategy: Find all function definitions that look like the benchmark
        # A definition has a body (i.e., followed by { )
        # Pattern: void funcname(params) {
        # We prioritize names containing the benchmark name or common suffixes
        
        # Generic pattern for function definitions
        pattern = r'void\s+(\w+)\s*\([^)]*\)\s*\{'
        
        matches = re.findall(pattern, code)
        
        if not matches:
            return default_func
        
        # Prioritize: exact benchmark name > name with _opt suffix > name with benchmark substring > first match
        for m in matches:
            if m.lower() == benchmark_name.lower():
                return m
        for m in matches:
            if m.lower() == f'{benchmark_name}_opt'.lower() or m.lower() == f'{benchmark_name}_optimized'.lower():
                return m
        for m in matches:
            if benchmark_name.lower() in m.lower():
                return m
        
        # Fallback to first match
        return matches[0]

    def evaluate(self, code: str, metadata: Dict[str, Any], **kwargs) -> EvalMetric:
        benchmark_name = metadata.get("benchmark_name")
        config = self.harness_module.get_benchmark_config(benchmark_name)
        
        if not config:
            return EvalMetric(compile_success=False, error=f"Unsupported standalone benchmark: {benchmark_name}")
        
        # Round 17: Prompt now requires LLM to keep the original function name.
        # Use benchmark_name directly as the function name.
        # Round 19.5: Check for reserved symbols (e.g. memcpy) and map to safe name.
        reserved_map = getattr(self.harness_module, 'RESERVED_SYMBOL_MAP', {})
        func_name = reserved_map.get(benchmark_name, benchmark_name)
        
        logger.info(f"[{benchmark_name}] Using function name: {func_name}")
        
        # Generate full code using Header + LLM Code + Footer
        full_code = self.harness_module.generate_full_code(benchmark_name, code, func_name)
        if not full_code:
            return EvalMetric(compile_success=False, error=f"Failed to generate harness for {benchmark_name}")


        # 3. Compile & Run (Delegate to Parent)
        kwargs['compile_timeout'] = 60 
        
        metric = super().evaluate(full_code, metadata, **kwargs)
        
        # 4. Post-Process Metric (Parse Execution Time from Output)
        if metric.compile_success and metric.correct:
            match = re.search(r'Average Execution Time:\s*([\d\.]+)\s*ms', metric.output)
            if match:
                try:
                    ms_time = float(match.group(1))
                    metric.time = ms_time / 1000.0
                except ValueError:
                    pass
        
        return metric
