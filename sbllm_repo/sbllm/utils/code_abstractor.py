import ast
import tokenize
import builtins
import logging
from io import BytesIO
from tree_sitter import Language, Parser

logger = logging.getLogger(__name__)

TREE_SITTER_DIR = './build/my-languages.so'

class CodeAbstractorPy(ast.NodeTransformer):
    def __init__(self, variable_char='VAR', string_char='STR', number_char='NUM'):
        self.variable_char = variable_char
        self.string_char = string_char
        self.number_char = number_char
        self.variable_count = 0
        self.string_count = 0
        self.number_count = 0
        self.builtin_functions = set(dir(builtins))

    def visit_FunctionDef(self, node):
        for arg in node.args.args:
            arg.arg = f'{self.variable_char}'
            self.variable_count += 1
        self.generic_visit(node)
        return node

    def visit_Constant(self, node):
        self.generic_visit(node)
        if isinstance(node.value, str):
            node.value = f'{self.string_char}'
            self.string_count += 1
        elif isinstance(node.value, (int, float, complex)):
            node.value = f'{self.number_char}'
            self.number_count += 1
        return node

    def visit_Name(self, node):
        self.generic_visit(node)
        if node.id not in self.builtin_functions:
            node.id = f'{self.variable_char}'
            self.variable_count += 1
        return node

    def visit_Str(self, node):
        self.generic_visit(node)
        node.s = f'{self.string_char}'
        self.string_count += 1
        return node

    def visit_Num(self, node):
        self.generic_visit(node)
        node.n = f'{self.number_char}'
        self.number_count += 1
        return node

def abstract_py_code(code):
    parsed_code = ast.parse(code)
    abstractor = CodeAbstractorPy()
    abstracted_ast = abstractor.visit(parsed_code)
    abstracted_code = ast.unparse(abstracted_ast)
    return abstracted_code


class CodeAbstractorCpp:
    def __init__(self, variable_char='VAR', string_char='STR', number_char='NUM'):
        self.variable_char = variable_char
        self.string_char = string_char 
        self.number_char = number_char
        self.variable_count = 0
        self.string_count = 0
        self.number_count = 0
        self.modified_texts = {} 
        
        self.parser = Parser()
        try:
            import tree_sitter_cpp
            # New API (tree-sitter >= 0.22)
            self.parser.language = Language(tree_sitter_cpp.language())
        except (ImportError, TypeError):
            try:
                # Fallback to old API (tree-sitter < 0.22)
                # In tree-sitter 0.22+, Language(path, name) raises TypeError (1 arg expected)
                CPP_LANGUAGE = Language(TREE_SITTER_DIR, 'cpp')
                self.parser.language = CPP_LANGUAGE
            except Exception as e:
                # Disable abstraction if loading fails
                logger.warning(f"Failed to load Tree-Sitter C++ language: {e}. Abstraction disabled.")
                self.parser = None

        self.cpp_keywords = {
    "alignas", "alignof", "asm", "auto", "include",
    "bool", "break", "case", "catch", "char",
    "char8_t", "char16_t", "char32_t", "class", "concept", "cout", "cin",
    "const", "consteval", "constexpr", "constinit", "const_cast",
    "continue", "co_await", "co_return", "co_yield", "decltype",
    "default", "delete", "do", "double", "dynamic_cast",
    "else", "enum", "explicit", "export", "extern", "endl",
    "false", "float", "for", "friend", "goto",  "main",
    "if", "inline", "int", "long", "mutable",
    "namespace", "new", "noexcept", "nullptr", "operator",
    "private", "protected", "public", "register", "reinterpret_cast",
    "requires", "return", "short", "signed", "sizeof", "std",
    "static", "static_assert", "static_cast", "struct", "switch",
    "template", "this", "thread_local", "throw", "true",
    "try", "typedef", "typeid", "typename", "union",
    "unsigned", "using", "virtual", "void", "volatile",
    "wchar_t", "while", "algorithm", "array", "atomic", "bitset", "chrono",
    "codecvt", "complex", "deque", "exception", "fstream",
    "functional", "future", "initializer_list", "iomanip", "ios",
    "iosfwd", "iostream", "istream", "iterator", "limits",
    "list", "locale", "map", "memory", "mutex",
    "new", "numeric", "optional", "ostream", "queue",
    "random", "ratio", "regex", "scoped_allocator", "set",
    "shared_mutex", "sstream", "stack", "stdexcept", "streambuf",
    "string", "string_view", "strstream", "system_error", "thread",
    "tuple", "type_traits", "typeindex", "typeinfo", "unordered_map",
    "unordered_set", "utility", "valarray", "variant", "vector",
    "version", "numbers", "ranges", "span", "bit", "concepts", "coroutine", "format", 
    "source_location", "syncstream", "any", "filesystem", "memory_resource", "execution",
    "optional", "string_view", "variant", "compare", "span", "cerr", "clog", "getline",
    "stoi", "stol", "stoul", "stoll", "stoull", "stof",
    "stod", "stold", "to_string", "to_wstring", "push_back", "pop_back",
    "emplace_back", "size", "resize", "empty", "at", "clear", "insert", "erase",
    "begin", "end", "sort", "find", "find_if", "count", "count_if",
    "replace", "replace_if", "copy", "copy_if", "remove", "remove_if",
    "reverse", "max_element", "min_element", "accumulate", "next_permutation",
    "pow", "sqrt", "sin", "cos", "tan", "asin", "acos", "atan",
    "atan2", "exp", "log", "log10", "ceil", "floor", "round", "fmod",
    "abs", "make_shared", "make_unique", "shared_ptr", "unique_ptr",
    "weak_ptr", "fstream", "ifstream", "ofstream", "stringstream", "swap", "pair",
    "make_pair", "bitset", "tuple", "make_tuple", "tie", "get", "map", "set", "unordered_map",
    "unordered_set", "multimap", "multiset", "queue", "priority_queue", "stack",
    "initializer_list", "hash", "function", "bind", "thread", "mutex", "lock_guard",
    "unique_lock", "condition_variable", "async", "launch", "future", "promise"
    }
    def abstract_code(self, source_code):
        if self.parser is None:
            # Fallback: return original code if parser failed to load
            return source_code

        self.modified_texts.clear() 
        self.source_code = source_code 
        tree = self.parser.parse(bytes(source_code, 'utf8'))
        root_node = tree.root_node
        self._traverse_node(root_node)
        return self._get_code_from_tree(root_node)
    
    def _traverse_node(self, node):
        if node.type == 'identifier':
            name = self._get_node_text(node)
            if not self._is_cpp_keyword(name):
                self.modified_texts[node.id] = self.variable_char
                self.variable_count += 1
                
        elif node.type == 'string_literal':
            self.modified_texts[node.id] = f'"{self.string_char}"'
            self.string_count += 1
            
        elif node.type in ('number_literal', 'numeric_literal'):
            self.modified_texts[node.id] = self.number_char
            self.number_count += 1
            
        elif node.type == 'char_literal':
            self.modified_texts[node.id] = f"'{self.string_char}'"
            self.string_count += 1
            
        for child in node.children:
            self._traverse_node(child)
    
    def _get_node_text(self, node):
        start_byte = node.start_byte
        end_byte = node.end_byte
        return self.source_code[start_byte:end_byte]
    
    def _get_code_from_tree(self, node):
        if not node.children:
            return self.modified_texts.get(node.id, self._get_node_text(node))
            
        result = []
        prev_child = None
        
        for child in node.children:
            if prev_child:
                if child.start_point[0] > prev_child.end_point[0]:
                    newlines = child.start_point[0] - prev_child.end_point[0]
                    indent = ' ' * child.start_point[1]
                    result.append('\n' * newlines + indent)
                elif child.start_point[0] == prev_child.end_point[0]:
                    result.append(' ')
            
            result.append(self._get_code_from_tree(child))
            prev_child = child
            
        return ''.join(result)
    
    def _is_cpp_keyword(self, name):
        return name in self.cpp_keywords or '::' in name

def abstract_cpp_code(cpp_code):
    abstractor = CodeAbstractorCpp()
    abstracted = abstractor.abstract_code(cpp_code)
    
    return abstracted


def tokenize_py_code(code):
    tokens = []
    g = tokenize.tokenize(BytesIO(code.encode('utf-8')).readline)
    for toknum, tokval, _, _, _ in g:
        if toknum != tokenize.ENCODING:
            tokens.append(tokval)
    return tokens

def tokenize_cpp_code(code):
    tokens = []
    code_bytes = code.encode('utf-8')
    stream = BytesIO(code_bytes)
    g = tokenize.tokenize(stream.readline)
    for toknum, tokval, _, _, _ in g:
        tokens.append(tokval)
    return tokens
