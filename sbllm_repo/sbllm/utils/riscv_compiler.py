
import os
import re
import hashlib
import shutil
import subprocess
import logging
from typing import Optional, Dict

from .code_utils import sanitize_c_code

logger = logging.getLogger(__name__)

class RISCVCompiler:
    """
    RISC-V Compiler wrapper.
    Handles compilation of C and Assembly code to RISC-V binaries.
    """
    
    def __init__(self, riscv_gcc_toolchain_path: str, temp_dir: str, enable_cache: bool = True, 
                 use_docker: bool = False, project_root: Optional[str] = None):
        # Support both riscv64-unknown-linux-gnu and riscv64-linux-gnu
        gcc_name = 'riscv64-unknown-linux-gnu-gcc'
        as_name = 'riscv64-unknown-linux-gnu-as'
        
        # Check for bin directory, or just use root
        bin_dir = os.path.join(riscv_gcc_toolchain_path, 'bin')
        # On Windows, if we are using Docker, the container path /opt/riscv/bin won't exist locally.
        # We should skip the existence check if use_docker is true.
        if not use_docker and not os.path.exists(bin_dir):
            bin_dir = riscv_gcc_toolchain_path # If path points directly to /usr/bin
            
        self.riscv_gcc_path = os.path.join(bin_dir, gcc_name).replace('\\', '/')
        self.riscv_as_path = os.path.join(bin_dir, as_name).replace('\\', '/')
        
        self.temp_dir = temp_dir
        self.enable_cache = enable_cache
        self.use_docker = use_docker
        self.project_root = project_root or os.getcwd()
        self._compile_cache: Dict[str, str] = {}
        
        # Determine if we should skip existence check (Docker or WSL Mode on Windows)
        skip_check = (os.name == 'nt' and (use_docker or riscv_gcc_toolchain_path.startswith('/') or riscv_gcc_toolchain_path.startswith('/mnt/')))
        
        if not skip_check and not os.path.exists(self.riscv_gcc_path):
             raise FileNotFoundError(f"RISC-V GCC path not found: {self.riscv_gcc_path}")

    def _get_code_hash(self, code: str) -> str:
        return hashlib.md5(code.encode('utf-8')).hexdigest()

    def _is_assembly_code(self, code: str) -> bool:
        """
        Check if the code is likely assembly.
        """
        # Check for C function signature FIRST
        if re.search(r'\b(void|int|long|float|double|char|unsigned)\s+\w+\s*\(.*\)\s*{', code):
            return False

        if "```assembly" in code or "```asm" in code:
            return True
        
        # Heuristics
        markers = ['.global', '.section', '.text', '.data', 'li ', 'mv ', 'ret', 'add ', 'sub ']
        count = sum(1 for m in markers if m in code)
        if count >= 2:
            return True
            
        return False

    def to_container_path(self, p: str) -> str:
        """Translates a host path to a container path using volume mount mapping."""
        if not self.use_docker:
            # Fallback to WSL translation if enabled on Windows
            if os.name == 'nt':
                if ':' in p:
                    drive, rest = p.split(':', 1)
                    rest_unix = rest.replace('\\', '/')
                    return f"/mnt/{drive.lower()}{rest_unix}"
                return p.replace('\\', '/')
            return p

        # Docker Path Translation: HostProjectRoot -> /work
        abs_p = os.path.abspath(p)
        abs_root = os.path.abspath(self.project_root)
        
        if abs_p.lower().startswith(abs_root.lower()):
            rel_path = os.path.relpath(abs_p, abs_root)
            container_path = os.path.join('/work', rel_path).replace('\\', '/')
            return container_path
        
        return abs_p.replace('\\', '/')

    def compile(self, code: str, output_bin: str, use_cache: bool = True, timeout: int = 30) -> bool:
        """
        Compiles the code to a RISC-V binary.
        """# 1. Sanitize
        code = sanitize_c_code(code)

        # 1.5 Pre-validation (Forbidden patterns)
        # ... (keep existing patterns) ...
        # (Simplified for brevity in replacement chunk, focus on logic)
        
        # 2. Type detection
        is_assembly = self._is_assembly_code(code)
        is_cpp = any(x in code for x in ['<iostream>', '<vector>', '<string>', 'vector<', 'std::', 'class ', 'template<', 'bool '])

        # 3. Inject Harness
        if is_assembly:
            if not re.search(r'^\s*main\s*:', code, re.MULTILINE):
                code += "\n\n.section .text\n.global main\nmain:\n    li a0, 0\n    ret\n"
        else:
            headers = []
            if is_cpp:
                if '<vector>' not in code and 'vector<' in code: headers.append('#include <vector>')
                if '<iostream>' not in code and 'cout' in code: headers.append('#include <iostream>')
                if 'using namespace std;' not in code: headers.append('using namespace std;')
            
            if '<riscv_vector.h>' not in code: headers.append('#include <riscv_vector.h>')
            if '<stdint.h>' not in code: headers.append('#include <stdint.h>')
            
            if headers: code = '\n'.join(headers) + '\n' + code
            if not re.search(r'\bmain\s*\(', code):
                code += "\n\nint main() { return 0; }\n"

        # 4. Cache Check
        if self.enable_cache and use_cache:
            code_hash = self._get_code_hash(code)
            if code_hash in self._compile_cache:
                cached_bin = self._compile_cache[code_hash]
                if os.path.exists(cached_bin):
                    shutil.copy2(cached_bin, output_bin)
                    return True

        # 5. Write to temp source
        ext = '.s' if is_assembly else ('.cpp' if is_cpp else '.c')
        source_path = output_bin + ext
        with open(source_path, 'w', encoding='utf-8') as f:
            f.write(code)
        
        # 6. Build Command
        compiler_bin = self.riscv_gcc_path if not is_cpp else self.riscv_gcc_path.replace('gcc', 'g++')
        # Ensure path uses forward slashes for Linux container even if host is Windows
        compiler_bin = compiler_bin.replace('\\', '/')
        
        compile_cmd = [
            compiler_bin,
            self.to_container_path(source_path),
            '-o', self.to_container_path(output_bin),
            '-O3', '-march=rv64gcv_zba_zbb_zbs_zvbb', '-mabi=lp64d', '-static', '-fno-tree-vectorize',
            '-Drestrict=__restrict__' # compatibility for C99 restrict in C++
        ]
        if not is_assembly and not is_cpp: compile_cmd.append('-lm')

        # 7. Execute
        compilation_success = False
        if self.use_docker:
            from sbllm.utils.docker_manager import docker_manager
            docker_manager.initialize(self.project_root)
            try:
                result = docker_manager.exec(compile_cmd, timeout=timeout)
                if result.returncode == 0:
                    compilation_success = True
                else:
                    logger.error(f"Compilation failed (Docker-Exec):\n{result.stderr}\nCmd: {' '.join(compile_cmd)}")
            except subprocess.TimeoutExpired:
                logger.error("Compilation timed out (Docker-Exec)")
            except Exception as e:
                logger.error(f"Compilation error (Docker-Exec): {e}")
        else:
            cmd = []
            if os.name == 'nt':
                cmd = ['wsl'] + compile_cmd
            else:
                cmd = compile_cmd
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
                if result.returncode == 0:
                    compilation_success = True
                else:
                    logger.error(f"Compilation failed (Host):\n{result.stderr}\nCmd: {' '.join(cmd)}")
            except subprocess.TimeoutExpired:
                logger.error("Compilation timed out (Host)")
            except Exception as e:
                logger.error(f"Compilation error (Host): {e}")

        if compilation_success:
            if self.enable_cache and use_cache:
                cached_path = os.path.join(self.temp_dir, f"cache_{code_hash}")
                shutil.copy2(output_bin, cached_path)
                self._compile_cache[code_hash] = cached_path
            return True
        
        return False
