#!/bin/bash
set -e

echo "Installing Python dependencies..."
echo "This may take a while (downloading PyTorch ~900MB)..."

# Optional: Use mirror if download is slow
# pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip3 install --no-cache-dir -r requirement.txt

echo "Setup complete!"
python3 -c "import torch; print(f'Torch {torch.__version__} installed')"
