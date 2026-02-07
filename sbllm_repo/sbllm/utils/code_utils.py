import re
import io
import tokenize

def remove_comments_and_docstrings(source: str, lang: str) -> str:
    """
    Remove comments and docstrings from source code.
    Currently supports python and c/cpp (basic support).
    """
    if lang == 'python':
        io_obj = io.BytesIO(source.encode('utf-8'))
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        try:
            for tok in tokenize.tokenize(io_obj.readline):
                token_type = tok.type
                token_string = tok.string
                start_line, start_col = tok.start
                end_line, end_col = tok.end
                if start_line > last_lineno:
                    last_col = 0
                if start_col > last_col:
                    out += (" " * (start_col - last_col))
                # Remove comments and docstrings
                if token_type == tokenize.COMMENT:
                    pass
                elif token_type == tokenize.STRING:
                    if prev_toktype != tokenize.INDENT:
                        if prev_toktype != tokenize.NEWLINE:
                            if start_col > 0:
                                out += token_string
                else:
                    out += token_string
                prev_toktype = token_type
                last_col = end_col
                last_lineno = end_line
        except tokenize.TokenError:
            return source
        return out
    elif lang in ['c', 'cpp', 'riscv']:
        # Basic comment removal using regex
        if lang == 'riscv':
            # Handles # comments
            pattern = r'#.*?$|//.*?$|/\*.*?\*/'
        else:
            # Handles // and /* */
            pattern = r'//.*?$|/\*.*?\*/'
        return re.sub(pattern, '', source, flags=re.MULTILINE|re.DOTALL)
    return source

def extract_code_from_markdown(text: str, lang: str = 'c') -> str:
    """
    Consolidated code extraction from markdown responses.
    """
    if not text:
        return ""
    
    # Check for triple backticks
    if '```' not in text:
        # Heuristic check: is the text itself code?
        if any(kw in text for kw in ['#include', 'int ', 'void ', 'asm ', 'def ', 'import ']):
            return text.strip()
        return text.strip()

    # Find all code blocks with optional language tags
    # Matches ```lang\n<code>``` or ```\n<code>```
    blocks = re.findall(r'```(?:\w+\s+)?(.*?)\n?(?=```)```', text, re.DOTALL)
    
    if not blocks:
        return text.strip()

    def score_block(code):
        score = len(code)
        # Higher score for language-specific features
        if any(kw in code for kw in ['#include', 'asm volatile', 'std::vector', 'vfloat32']):
            score += 1000
        if '{' in code and '}' in code:
            score += 500
        # Penalty for explanation text inside block
        if 'Here is' in code or 'optimized' in code.lower():
            score -= 2000
        return score

    # Return the block with the highest heuristic score
    return max(blocks, key=score_block).strip()

def validate_c_syntax_heuristic(code: str) -> bool:
    """
    Heuristic validation for C code.
    1. Checks for non-ASCII characters (critical for filtering Chinese text).
    2. Checks for basic structural integrity (semicolons, braces).
    3. Checks for common hallucination patterns (double type decls).
    """
    if not code or not code.strip():
        return False
        
    # CRITICAL: Reject non-ASCII characters
    try:
        code.encode('ascii')
    except UnicodeEncodeError:
        return False

    # Check for basic C structure
    if ';' not in code and '}' not in code and '#include' not in code:
        # Assembly might not have semicolons but usually has labels/directives
        # If it's pure C, it needs these.
        # Let's be lenient for assembly if lang is riscv, but strict for C
        return False
        
    # Check for "int .* = int " hallucination
    if re.search(r'\b(int|long|float|double|char|void)\s+\w+\s*=\s*\b(int|long|float|double|char|void)\s+', code):
        return False
        
    return True

def sanitize_c_code(code: str) -> str:
    """
    Clean up C code: remove comments, strip whitespace, auto-inject missing headers.
    """
    code = remove_comments_and_docstrings(code, 'c')
    
    # Helper function to inject header
    def inject_header(code: str, header: str) -> str:
        if header not in code:
            if '#include' in code:
                code = code.replace('#include', f'#include {header}\n#include', 1)
            else:
                code = f'#include {header}\n' + code
        return code
    
    # =========================================================================
    # <math.h> - Mathematical functions
    # =========================================================================
    math_funcs = [
        'pow(', 'sqrt(', 'fabs(', 'ceil(', 'floor(', 'round(',
        'sin(', 'cos(', 'tan(', 'asin(', 'acos(', 'atan(', 'atan2(',
        'exp(', 'log(', 'log2(', 'log10(', 'fmaf(', 'fma(',
        'hypot(', 'cbrt(', 'fmod(', 'remainder(', 'copysign(',
        'fmin(', 'fmax(', 'fdim(', 'trunc(', 'nearbyint('
    ]
    if any(func in code for func in math_funcs):
        code = inject_header(code, '<math.h>')
    
    # =========================================================================
    # <stdint.h> - Fixed-width integer types
    # =========================================================================
    int_types = [
        'uint8_t', 'uint16_t', 'uint32_t', 'uint64_t',
        'int8_t', 'int16_t', 'int32_t', 'int64_t',
        'uintptr_t', 'intptr_t', 'size_t', 'ptrdiff_t',
        'INT8_MAX', 'INT16_MAX', 'INT32_MAX', 'INT64_MAX',
        'UINT8_MAX', 'UINT16_MAX', 'UINT32_MAX', 'UINT64_MAX'
    ]
    if any(t in code for t in int_types):
        code = inject_header(code, '<stdint.h>')
    
    # =========================================================================
    # <stdlib.h> - General utilities
    # =========================================================================
    stdlib_funcs = [
        'abs(', 'labs(', 'llabs(', 'malloc(', 'free(', 'calloc(', 'realloc(',
        'atoi(', 'atol(', 'atoll(', 'strtol(', 'strtoll(', 'strtoul(',
        'rand(', 'srand(', 'exit(', 'abort(', 'qsort(', 'bsearch(',
        'div(', 'ldiv(', 'getenv('
    ]
    if any(func in code for func in stdlib_funcs):
        code = inject_header(code, '<stdlib.h>')
    
    # =========================================================================
    # <string.h> - String handling
    # =========================================================================
    string_funcs = [
        'memcpy(', 'memmove(', 'memset(', 'memcmp(', 'memchr(',
        'strcpy(', 'strncpy(', 'strcat(', 'strncat(',
        'strcmp(', 'strncmp(', 'strchr(', 'strrchr(', 'strstr(',
        'strlen(', 'strtok(', 'strdup(', 'strerror('
    ]
    if any(func in code for func in string_funcs):
        code = inject_header(code, '<string.h>')
    
    # =========================================================================
    # <stdio.h> - Input/output
    # =========================================================================
    stdio_funcs = [
        'printf(', 'fprintf(', 'sprintf(', 'snprintf(',
        'scanf(', 'fscanf(', 'sscanf(',
        'fopen(', 'fclose(', 'fread(', 'fwrite(',
        'fgets(', 'fputs(', 'puts(', 'getchar(', 'putchar(',
        'FILE', 'stdin', 'stdout', 'stderr', 'EOF'
    ]
    if any(func in code for func in stdio_funcs):
        code = inject_header(code, '<stdio.h>')
    
    # =========================================================================
    # <stdbool.h> - Boolean type
    # =========================================================================
    if 'bool ' in code or ' bool' in code or 'true' in code or 'false' in code:
        if '_Bool' not in code:  # _Bool is built-in, no header needed
            code = inject_header(code, '<stdbool.h>')
    
    # =========================================================================
    # <limits.h> - Implementation limits
    # =========================================================================
    limits_macros = [
        'CHAR_BIT', 'CHAR_MAX', 'CHAR_MIN',
        'INT_MAX', 'INT_MIN', 'LONG_MAX', 'LONG_MIN',
        'SHRT_MAX', 'SHRT_MIN', 'UCHAR_MAX', 'UINT_MAX', 'ULONG_MAX'
    ]
    if any(m in code for m in limits_macros):
        code = inject_header(code, '<limits.h>')
    
    # =========================================================================
    # <float.h> - Floating-point limits
    # =========================================================================
    float_macros = [
        'FLT_MAX', 'FLT_MIN', 'FLT_EPSILON',
        'DBL_MAX', 'DBL_MIN', 'DBL_EPSILON'
    ]
    if any(m in code for m in float_macros):
        code = inject_header(code, '<float.h>')
    
    # =========================================================================
    # <assert.h> - Assertions
    # =========================================================================
    if 'assert(' in code:
        code = inject_header(code, '<assert.h>')
    
    # =========================================================================
    # <ctype.h> - Character handling
    # =========================================================================
    ctype_funcs = [
        'isalpha(', 'isdigit(', 'isalnum(', 'isspace(',
        'isupper(', 'islower(', 'toupper(', 'tolower('
    ]
    if any(func in code for func in ctype_funcs):
        code = inject_header(code, '<ctype.h>')
    
    # =========================================================================
    # <time.h> - Time handling
    # =========================================================================
    time_funcs = [
        'time(', 'clock(', 'difftime(', 'mktime(',
        'strftime(', 'localtime(', 'gmtime(',
        'time_t', 'clock_t', 'struct tm'
    ]
    if any(func in code for func in time_funcs):
        code = inject_header(code, '<time.h>')
    
    if any(func in code for func in time_funcs):
        code = inject_header(code, '<time.h>')
    
    return code.strip()

def extract_py(code, text):
    """Wrapper for extract_code_from_markdown for Python"""
    return extract_code_from_markdown(text, lang='python')

def extract_cpp(code, text):
    """Wrapper for extract_code_from_markdown for C/C++/RISC-V"""
    # code arg is unused but kept for compatibility with call signature
    return extract_code_from_markdown(text, lang='riscv')
