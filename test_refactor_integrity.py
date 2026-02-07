import sys
import os
import unittest
from unittest.mock import MagicMock

# 确保导入路径正确
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "sbllm_repo")))

class TestSBLLMRefactor(unittest.TestCase):
    def test_arg_parser_sync(self):
        """测试统一参数解析器是否包含所有核心参数"""
        from sbllm.core.arg_parser import cfg_parsing
        
        # 模拟命令行参数
        test_args = [
            "prog", "--mode", "test", "--lang", "riscv", "--model_name", "gpt-4",
            "--output_path", "./output", "--generation_path", "./gen",
            "--training_data_path", "./train", "--iteration", "2"
        ]
        import sys
        original_argv = sys.argv
        sys.argv = test_args
        
        try:
            cfg = cfg_parsing()
            self.assertEqual(cfg.mode, "test")
            self.assertEqual(cfg.iteration, 2)
            # 验证新增的参数是否存在
            self.assertTrue(hasattr(cfg, "retrieval_method"))
            self.assertTrue(hasattr(cfg, "beam_number"))
        finally:
            sys.argv = original_argv

    def test_code_utils_extraction(self):
        """测试整合后的代码提取工具"""
        from sbllm.utils.code_utils import extract_code_from_markdown
        
        md_text = "Here is the code:\n```c\nvoid foo() {}\n```\nExplanation."
        extracted = extract_code_from_markdown(md_text, lang='c')
        self.assertEqual(extracted, "void foo() {}")
        
        # 测试混合内容提取
        mixed_text = "Standard markdown.\n```riscv\nli a0, 1\n```"
        extracted_rv = extract_code_from_markdown(mixed_text, lang='riscv')
        self.assertEqual(extracted_rv, "li a0, 1")

    def test_module_imports(self):
        """验证重构后的核心模块是否可正常导入"""
        try:
            import sbllm.evol_query
            import sbllm.merge
            import sbllm.execution
            print("All core modules imported successfully.")
        except ImportError as e:
            self.fail(f"Module import failed: {e}")

if __name__ == "__main__":
    unittest.main()
