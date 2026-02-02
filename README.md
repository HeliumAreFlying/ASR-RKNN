# ASR-RKNN

基于TDNN（Time Delay Neural Network）的中文自动语音识别系统，支持RKNN部署优化。

## 项目简介

本项目实现了一个轻量级的中文语音识别系统，采用TDNN架构和CTC损失函数进行端到端训练。模型设计考虑了边缘设备部署需求，支持RKNN格式转换以在瑞芯微芯片上高效运行。

## 主要特性

- **TDNN架构**: 使用时延神经网络处理语音序列，具有良好的时序建模能力
- **CTC训练**: 采用CTC（Connectionist Temporal Classification）损失函数，无需对齐标注
- **多语言支持**: 主要针对中文优化，同时支持多种语言的音频样例
- **边缘部署**: 支持RKNN格式转换，适合在资源受限的设备上运行
- **滑窗推理**: 实现了高效的滑窗推理机制，支持长音频处理

## 项目结构

```
ASR-RKNN/
├── examples/           # 音频样例文件
│   ├── en.wav         # 英文样例
│   ├── ja.wav         # 日文样例
│   ├── ko.wav         # 韩文样例
│   ├── yue.wav        # 粤语样例
│   └── zh.wav         # 中文样例
├── training/          # 训练相关代码
│   ├── main_model.py  # TDNN模型定义
│   └── train_entry.py # 训练入口脚本
├── utils/             # 工具函数
│   ├── basic_tools.py      # 基础工具函数
│   ├── get_feat.py         # 特征提取
│   └── get_vocab_and_meta.py # 词汇表和元数据处理
├── validation/        # 验证和推理
│   └── inference.py   # 推理脚本
└── README.md
```

## 环境要求

### Python依赖
```bash
torch>=1.9.0
torchaudio
librosa
soundfile
kaldi-native-fbank
torchinfo
numpy
```

### 安装依赖
```bash
pip install torch torchaudio librosa soundfile kaldi-native-fbank torchinfo numpy
```

## 快速开始

### 1. 数据准备

准备训练数据，数据格式应为：
- 音频文件：WAV格式，16kHz采样率
- 元数据文件：CSV格式，包含音频路径和对应文本标注

### 2. 词汇表生成

```bash
cd utils
mkdir ../basic_data
python get_vocab_and_meta.py
```

这将生成：
- `vocab_data.json`: 词汇表文件
- `clean_meta_data.json`: 清理后的元数据文件

### 3. 模型训练

```bash
python training/train_entry.py
```

训练参数可在 `train_entry.py` 中修改：
- `BATCH_SIZE`: 批次大小（默认64）
- `LEARNING_RATE`: 学习率（默认1e-3）
- `NUM_EPOCHS`: 训练轮数（默认10000）

### 4. 模型推理

```python
from training.main_model import TDNNASR
import json

# 加载词汇表
vocab_data = json.load(open("basic_data/vocab_data.json"))

# 创建模型
model = TDNNASR(
    input_dim=560,
    block_dims=[512] * 9,
    dilations=[1, 2, 4, 2, 1, 2, 4, 2, 1],
    strides=[1, 1, 1, 1, 1, 1, 1, 1, 2],
    proj_dim=128,
    num_classes=vocab_data['vocab_size'] + 1,
    vocab_data=vocab_data
)

# 加载权重
model.load_state_dict(torch.load("weights/best.pth"))

# 推理
output, sentence = model.forward_wave("examples/zh.wav", need_sentence=True)
print(f"识别结果: {sentence}")
```

## 模型架构

### TDNN网络结构
- **输入维度**: 560维特征（80维Fbank × 7帧窗口）
- **网络层数**: 9层残差TDNN块
- **膨胀率**: [1, 2, 4, 2, 1, 2, 4, 2, 1]
- **步长**: [1, 1, 1, 1, 1, 1, 1, 1, 2]
- **输出层**: 投影层 + 分类层

### 特征提取
- **音频预处理**: 16kHz重采样
- **特征类型**: 80维Fbank特征
- **窗口设置**: 7帧窗口，1帧步长
- **最终特征**: 560维（80×7）

## 训练配置

### 数据配置
- 训练/验证集划分：9:1
- 批次大小：64
- 数据增强：支持动态长度padding

### 优化器配置
- 优化器：AdamW
- 学习率：1e-3
- 学习率调度：ReduceLROnPlateau
- 梯度裁剪：最大范数5.0

### 损失函数
- CTC Loss（blank=0）
- 支持变长序列训练
- 零无穷大处理

## 部署说明

### RKNN转换
模型训练完成后，可转换为RKNN格式以在瑞芯微芯片上部署：

1. 导出ONNX模型
2. 使用RKNN-Toolkit转换为RKNN格式
3. 在目标设备上加载RKNN模型进行推理

### 性能优化
- 滑窗推理：支持长音频的分块处理
- 批量推理：支持多音频并行处理
- 内存优化：动态内存分配，避免OOM

## 许可证

本项目采用MIT许可证。

## 贡献

欢迎提交Issue和Pull Request来改进项目。

## 联系方式

如有问题或建议，请通过Issue联系。
