#!/usr/bin/env python3
"""
RISC-V 环境验证脚本

用于验证 RISC-V 优化框架运行所需的所有依赖和配置。
"""

import os
import sys
import jsonlines
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional


class Colors:
    """终端颜色 (Disabled for logging)"""
    GREEN = ''
    RED = ''
    YELLOW = ''
    BLUE = ''
    RESET = ''
    BOLD = ''


def print_success(message: str):
    """打印成功消息"""
    print(f"{Colors.GREEN}✓{Colors.RESET} {message}")


def print_error(message: str):
    """打印错误消息"""
    print(f"{Colors.RED}✗{Colors.RESET} {message}")


def print_warning(message: str):
    """打印警告消息"""
    print(f"{Colors.YELLOW}⚠{Colors.RESET} {message}")


def print_info(message: str):
    """打印信息消息"""
    print(f"{Colors.BLUE}ℹ{Colors.RESET} {message}")


def check_file_exists(file_path: str, description: str) -> bool:
    """检查文件是否存在"""
    if os.path.exists(file_path):
        print_success(f"{description}: {file_path}")
        return True
    else:
        print_error(f"{description} 不存在: {file_path}")
        return False


def check_header_file(include_dir: str, header_name: str, description: str) -> bool:
    """检查头文件是否存在"""
    header_path = os.path.join(include_dir, header_name)
    if os.path.exists(header_path):
        print_success(f"{description}: {header_path}")
        return True
        
    # Search recursively if not found in root include
    for root, dirs, files in os.walk(include_dir):
        if header_name in files:
            full_path = os.path.join(root, header_name)
            print_success(f"{description}: {full_path}")
            return True
            
    print_error(f"{description} 未找到: {header_name} (in {include_dir})")
    return False

def check_executable(exec_path: str, description: str) -> Tuple[bool, Optional[str]]:
    """检查可执行文件是否存在并可运行"""
    if not os.path.exists(exec_path):
        print_error(f"{description} 不存在: {exec_path}")
        return False, None
    
    if not os.access(exec_path, os.X_OK):
        print_error(f"{description} 不可执行: {exec_path}")
        return False, None
    
    try:
        # 尝试运行并获取版本信息
        result = subprocess.run(
            [exec_path, '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        version_info = result.stdout.strip().split('\n')[0] if result.returncode == 0 else None
        print_success(f"{description}: {exec_path}")
        if version_info:
            print_info(f"  版本信息: {version_info[:80]}")
        return True, version_info
    except subprocess.TimeoutExpired:
        print_warning(f"{description} 版本检查超时: {exec_path}")
        return True, None
    except Exception as e:
        print_warning(f"{description} 版本检查失败: {e}")
        return True, None


def check_directory(dir_path: str, description: str, create_if_missing: bool = False) -> bool:
    """检查目录是否存在"""
    if os.path.exists(dir_path):
        if os.path.isdir(dir_path):
            print_success(f"{description}: {dir_path}")
            return True
        else:
            print_error(f"{description} 不是目录: {dir_path}")
            return False
    else:
        if create_if_missing:
            try:
                os.makedirs(dir_path, exist_ok=True)
                print_success(f"{description} 已创建: {dir_path}")
                return True
            except Exception as e:
                print_error(f"无法创建 {description}: {e}")
                return False
        else:
            print_error(f"{description} 不存在: {dir_path}")
            return False


def check_jsonl_file(file_path: str, description: str, required_fields: List[str] = None) -> bool:
    """检查 JSONL 文件是否存在且格式正确"""
    if not check_file_exists(file_path, description):
        return False
    
    try:
        count = 0
        with jsonlines.open(file_path) as f:
            for obj in f:
                count += 1
                if required_fields:
                    for field in required_fields:
                        if field not in obj:
                            print_warning(f"{description} 中第 {count} 条记录缺少字段: {field}")
                            return False
                # 只检查前几条记录
                if count >= 10:
                    break
        
        if count == 0:
            print_warning(f"{description} 为空文件")
            return False
        
        print_success(f"{description} 格式正确，包含至少 {count} 条记录")
        return True
    except Exception as e:
        print_error(f"{description} 格式错误: {e}")
        return False


def check_api_keys(evol_query_file: str) -> bool:
    """检查 API 密钥是否配置"""
    try:
        with open(evol_query_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否有占位符 API 密钥
        placeholder_patterns = [
            'xxxxxxxxxxxxxxxxxxxx',
            'your_api_key_here',
            'api_key_placeholder'
        ]
        
        has_placeholder = any(pattern in content for pattern in placeholder_patterns)
        
        if has_placeholder:
            print_warning("检测到占位符 API 密钥，请确保已配置真实的 API 密钥")
            print_info("  需要在 evol_query.py 中配置以下 API 密钥:")
            print_info("    - openai_api_keys")
            print_info("    - gemini_api_keys")
            print_info("    - deepseek_api_keys")
            print_info("    - llama_api_keys")
            return False
        else:
            print_success("API 密钥配置检查通过（未检测到占位符）")
            return True
    except Exception as e:
        print_warning(f"无法检查 API 密钥配置: {e}")
        return False


def check_riscv_toolchain(toolchain_path: str) -> bool:
    """检查 RISC-V 工具链是否完整"""
    if not os.path.exists(toolchain_path):
        print_error(f"RISC-V 工具链路径不存在: {toolchain_path}")
        return False
    
    bin_dir = os.path.join(toolchain_path, 'bin')
    if not os.path.exists(bin_dir):
        print_error(f"RISC-V 工具链 bin 目录不存在: {bin_dir}")
        return False
    
    required_tools = [
        'riscv64-unknown-linux-gnu-gcc',
        'riscv64-unknown-linux-gnu-as',
        'riscv64-unknown-linux-gnu-objdump',
        'riscv64-unknown-linux-gnu-size'
    ]
    
    all_exist = True
    for tool in required_tools:
        tool_path = os.path.join(bin_dir, tool)
        if os.path.exists(tool_path):
            print_success(f"工具链工具: {tool}")
        else:
            print_error(f"工具链工具缺失: {tool}")
            all_exist = False
    
    # remove premature return
    # msg: Toolchain checking continues...

    include_dir = os.path.join(toolchain_path, 'lib', 'gcc', 'riscv64-unknown-linux-gnu', '14.2.0', 'include')
    # Fallback to general include search if specific path fails/varies
    if not os.path.exists(include_dir):
        # Try to find lib directory
         for root, dirs, files in os.walk(toolchain_path):
             if 'riscv_vector.h' in files:
                 include_dir = root
                 break
    
    if not check_header_file(include_dir, 'riscv_vector.h', 'RISC-V Vector Header'):
        all_exist = False

    return all_exist


def validate_riscv_setup(
    qemu_path: str,
    riscv_gcc_toolchain_path: str,
    data_dir: str = "../processed_data/riscv",
    output_dir: str = "../output/riscv",
    evol_query_file: str = "evol_query.py"
) -> bool:
    """
    验证 RISC-V 环境设置
    
    Args:
        qemu_path: QEMU 可执行文件路径
        riscv_gcc_toolchain_path: RISC-V GCC 工具链路径
        data_dir: 数据目录路径
        output_dir: 输出目录路径
        evol_query_file: evol_query.py 文件路径
    
    Returns:
        验证是否通过
    """
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}RISC-V 环境验证{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")
    
    all_checks_passed = True
    
    # 1. 检查 QEMU
    print(f"{Colors.BOLD}1. 检查 QEMU{Colors.RESET}")
    print("-" * 60)
    qemu_ok, _ = check_executable(qemu_path, "QEMU")
    if not qemu_ok:
        all_checks_passed = False
    print()
    
    # 2. 检查 RISC-V 工具链
    print(f"{Colors.BOLD}2. 检查 RISC-V GCC 工具链{Colors.RESET}")
    print("-" * 60)
    toolchain_ok = check_riscv_toolchain(riscv_gcc_toolchain_path)
    if not toolchain_ok:
        all_checks_passed = False
    print()
    
    # 3. 检查数据文件
    print(f"{Colors.BOLD}3. 检查数据文件{Colors.RESET}")
    print("-" * 60)
    test_data_path = os.path.join(data_dir, "test.jsonl")
    train_data_path = os.path.join(data_dir, "train.jsonl")
    
    test_ok = check_jsonl_file(
        test_data_path,
        "测试数据文件",
        required_fields=["idx", "query"]
    )
    if not test_ok:
        all_checks_passed = False
    
    train_ok = check_jsonl_file(
        train_data_path,
        "训练数据文件（知识库）",
        required_fields=["id", "original_code", "optimized_code"]
    )
    if not train_ok:
        all_checks_passed = False
    print()
    
    # 4. 检查输出目录
    print(f"{Colors.BOLD}4. 检查输出目录{Colors.RESET}")
    print("-" * 60)
    output_ok = check_directory(output_dir, "输出目录", create_if_missing=True)
    if not output_ok:
        all_checks_passed = False
    print()
    
    # 5. 检查 API 密钥
    print(f"{Colors.BOLD}5. 检查 API 密钥配置{Colors.RESET}")
    print("-" * 60)
    api_ok = check_api_keys(evol_query_file)
    if not api_ok:
        # 只发出警告，不阻止运行（允许 dry run）
        print_warning("API 密钥未配置，后续 LLM 调用可能会失败")
        # all_checks_passed = False  <-- Disabled strict check
    print()
    
    # 6. 检查 Python 依赖
    print(f"{Colors.BOLD}6. 检查 Python 依赖{Colors.RESET}")
    print("-" * 60)
    required_packages = [
        'jsonlines', 'numpy', 'openai', 'tqdm', 'google.generativeai',
        'rank_bm25', 'editdistance', 'tree_sitter', 'faiss', 
        'sentence_transformers', 'pandas'
    ]
    
    missing_packages = []
    optional_packages = ['faiss', 'sentence_transformers', 'google.generativeai']
    
    for package in required_packages:
        try:
            __import__(package.replace('.', '_') if '.' in package else package)
            print_success(f"Python 包: {package}")
        except ImportError:
            if package in optional_packages:
                print_warning(f"Python 包缺失 (可选): {package}")
            else:
                print_error(f"Python 包缺失: {package}")
                missing_packages.append(package)
    
    if missing_packages:
        all_checks_passed = False
        print_warning(f"请运行: pip install {' '.join(missing_packages)}")
    print()
    
    # 总结
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    if all_checks_passed:
        print(f"{Colors.BOLD}{Colors.GREEN}✓ 所有检查通过！环境配置正确。{Colors.RESET}")
        return True
    else:
        print(f"{Colors.BOLD}{Colors.RED}✗ 部分检查未通过，请修复上述问题后重试。{Colors.RESET}")
        print("Failed Checks Summary:")
        if not qemu_ok: print("- QEMU Check Failed")
        if not toolchain_ok: print("- Toolchain Check Failed")
        if not test_ok: print("- Test Data Check Failed")
        if not train_ok: print("- Train Data Check Failed")
        if not output_ok: print("- Output Directory Check Failed")
        if not api_ok: print("- API Key Check Failed (Warning)")
        if missing_packages: print(f"- Missing Packages: {missing_packages}")
        return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="验证 RISC-V 优化框架环境配置"
    )
    parser.add_argument(
        "--qemu_path",
        type=str,
        required=True,
        help="QEMU 可执行文件路径（如 /usr/bin/qemu-riscv64）"
    )
    parser.add_argument(
        "--riscv_gcc_toolchain_path",
        type=str,
        required=True,
        help="RISC-V GCC 工具链路径（如 /opt/riscv64-unknown-linux-gnu）"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../processed_data/riscv",
        help="数据目录路径（默认: ../processed_data/riscv）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../output/riscv",
        help="输出目录路径（默认: ../output/riscv）"
    )
    parser.add_argument(
        "--evol_query_file",
        type=str,
        default="evol_query.py",
        help="evol_query.py 文件路径（默认: evol_query.py）"
    )
    
    args = parser.parse_args()
    
    # 转换为绝对路径
    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir)
    evol_query_file = os.path.abspath(args.evol_query_file) if not os.path.isabs(args.evol_query_file) else args.evol_query_file
    
    success = validate_riscv_setup(
        qemu_path=args.qemu_path,
        riscv_gcc_toolchain_path=args.riscv_gcc_toolchain_path,
        data_dir=data_dir,
        output_dir=output_dir,
        evol_query_file=evol_query_file
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

