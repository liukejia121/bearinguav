

安装方式（强烈建议这样做）
1️⃣ 创建环境
conda create -n bearing_env python=3.9 -y
conda activate bearing_env
2️⃣ 安装 PyTorch（关键）
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu118 \
3️⃣ 安装其余依赖
pip install -r requirements.txt



pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu118 \
  --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn