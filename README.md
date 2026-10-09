# 放射性核素识别 · Gamma Spectrum CNN

使用 PyTorch 一维卷积神经网络，对 **1024 道 γ 射线能谱**进行分类。项目包含数据预处理、谱数据增强、模型训练与分类评估。

## 技术与流程

- **预处理**：`log1p` 变换与 Min-Max 归一化，输入形状为 `(batch, 1, 1024)`。
- **增强**：泊松重采样、增益漂移与道址平移，每类默认生成 180 个样本。
- **模型**：4 层 Conv1D（16 → 32 → 64 → 64），Dropout 与全连接分类头。
- **训练**：Adam、权重衰减、验证损失早停和学习率降低，自动选择 CPU / CUDA。
- **评估**：分类报告、混淆矩阵及逐样本预测 CSV。

## 项目结构

```text
src/
  model.py             # SpectralCNN 网络
  dataset_loader.py    # 数据读取、归一化与 DataLoader
  generate_data.py     # 从原始谱生成增强训练数据
  train.py             # 训练与训练曲线
  evaluate_model.py    # 验证集分类报告与混淆矩阵
  test.py              # 新数据推理与预测 CSV
  plot_generated.py    # 原始谱与增强谱对照
dataset/README.md      # 数据格式与目录约定
docs/整理说明.md       # 上传内容选择与已知限制
pyproject.toml         # 环境依赖
uv.lock                # 锁定的依赖解析
```

## 快速开始

推荐 Python 3.12，在**仓库根目录**执行：

```bash
uv sync --locked
```

按 [数据准备指南](dataset/README.md) 放置自己的 `.dat` 文件。本仓库不包含原始实验数据或预训练权重；训练前必须先准备数据。

```bash
uv run python src/generate_data.py
uv run python src/train.py
uv run python src/evaluate_model.py
```

训练保存 `best_model.pth` 和 `training_history.png`；评估保存 `confusion_matrix.png`。把独立测试数据放入 `dataset/New/<类别>/` 后，可运行：

```bash
uv run python src/test.py
```

## 当前实现的边界

- 训练自动扫描类别；`evaluate_model.py` 和 `test.py` 当前固定为 **Background、Cs137、I131**。默认流程请只准备这三类；扩展类别时必须同步更新评估脚本中的 `TRAIN_CLASSES`，顺序应与训练的排序一致。
- `plot_generated.py` 默认展示 Co57、K40；使用三类数据时，需要把末尾 `target_classes` 改为实际类别。
- 增强脚本每次重新生成 `dataset/generated/`，其中只应存放可重新生成的数据。
- 当前验证集使用生成训练数据的种子谱，二者具有来源关联；验证准确率不能直接代表独立测量上的泛化性能。正式评估应使用独立采集的数据。
- 数据增强与训练尚未统一设置随机种子，重复运行的结果可能不同。

仓库说明基于现有代码，不声明未经独立验证的准确率。
