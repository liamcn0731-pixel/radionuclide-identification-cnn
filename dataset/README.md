# 数据准备

原始实验数据保留在本地，不随仓库上传。准备数据时创建以下结构：

```text
dataset/
  seeds/
    Background/*.dat
    Cs137/*.dat
    I131/*.dat
  generated/           # generate_data.py 自动生成
  New/                 # 可选：独立测试数据，使用相同类别目录
    Background/*.dat
    Cs137/*.dat
    I131/*.dat
```

每个 `.dat` 文件末尾的 **4096 字节**应为 1024 个小端无符号 32 位计数值（`<u4`）。文件头可存在，但当前分类流程不使用其刻度或其他元数据。`.npy` 输入应包含长度为 1024 的非负计数数组。

每个类别目录需至少有一个有效样本。默认评估脚本要求上述三类，类别映射为 `Background=0, Cs137=1, I131=2`。

请先保留原始数据备份，再生成增强数据；增强脚本会清理并重建 `dataset/generated/`。
