# PLT-ETC

基于循环 Transformer（Looped Transformer）的加密流量分类系统。在 UER 风格的预训练框架之上，
将编码器替换为**参数共享的循环 Transformer**，并支持**动态早停（Adaptive Computation Time, ACT）**：
简单样本一两轮循环即可停机，困难样本多循环，`loop_num` 仅作为最大步数兜底，
从而用一套权重覆盖多档算力预算。

## 目录结构

```
traffictransformer/        # 基线系统（数据、词表、预训练、微调、uer 框架）
├── fine-data/             # 微调数据生成（pcap -> flow -> tsv）
├── vocab_gen/             # 流量特征提取与词表生成
├── pre-training/          # MLM 预训练（preprocess.py -> pretrain.py）
├── fine-tuning/           # 基线分类微调（run_classifier.py）
├── models/bert/           # bert 系列配置文件
├── uer/                   # 预训练框架源码
└── 运行流程.md             # 基线系统原始处理流程

loopedtransformer/         # 循环 Transformer 新增模块（本仓库核心增量）
├── build.py               # 向 uer 注册表运行时注入循环编码器（不改 uer 源码）
├── looped_encoder.py      # 参数共享循环编码器（基线）
├── halting_looped_encoder.py  # 动态停机循环编码器（ACT，halting_looped_transformer）
├── looped_model.py        # 循环编码器模型外壳（逐循环监督 + 一致性蒸馏）
├── looped_trainer.py      # 循环预训练 trainer
├── pretrain_looped.py     # 循环预训练入口
├── finetune_looped.py     # 循环微调入口（分类）
└── configs/               # 循环模型超参配置（如 looped_random_sup_config.json）
```

## 核心机制

1. **参数共享循环**：同一层 Transformer 权重循环执行 `loop_num` 次。
2. **动态早停（默认开启，`--dynamic_halting`）**：逐样本累计把握度超过
   `--halting_threshold`（默认 0.99）即停机；`--ponder_weight`（默认 0.01）为停机步数惩罚，
   迫使模型学会尽早停机；`--halting_pooling` 指定停机打分方式（默认跟随 `--pooling`）。
   停机判定复刻 Universal Transformer 官方实现（still_running / new_halted / remainders）。
3. **深度扫描评估**：`--eval_loop_nums 2,4,6,8,12` 测试阶段用同一套权重逐个循环深度评估，
   输出精度-深度对照表。
4. **逐循环监督 + 一致性蒸馏**：预训练由 `looped_model.py` 实现，配合
   `loop_sup_weight` / `loop_distill_weight` 控制权重（见 configs）。

> 注：早期实验中的弹性深度（elastic_loop）口径已弃用，相关代码与参数已移除。

## 环境安装

```bash
pip install -r requirements.txt
```

需要 GPU 环境（torch + CUDA）。

## 使用流程

### 1. 数据准备与词表（详见 traffictransformer/运行流程.md）

```bash
# pcap -> flow（Windows 下可用 fine-data/SplitCap.exe）
python3 traffictransformer/fine-data/get_finedata.py
# 流量特征提取 -> 词表 -> 预训练数据集
python3 traffictransformer/vocab_gen/extractIpTcp.py
python3 traffictransformer/vocab_gen/vocab_gen_all.py
python3 traffictransformer/pre-training/preprocess.py
```

### 2. 循环预训练

```bash
python3 loopedtransformer/pretrain_looped.py     --dataset_path traffictransformer/pre-training/dataset/dataset.pt     --vocab_path traffictransformer/vocab_gen/vocab_packet_4chars_all.txt     --config_path loopedtransformer/configs/looped_random_sup_config.json     --target bertpackets     --loop_num 12     --total_steps 120000
```

### 3. 分类微调

```bash
python3 loopedtransformer/finetune_looped.py     --train_path traffictransformer/fine-data/fine_data_flow/<dataset>_enhance1/0.8/train_500_dataset.tsv     --dev_path   traffictransformer/fine-data/fine_data_flow/<dataset>_enhance1/0.8/valid_500_dataset.tsv     --test_path  traffictransformer/fine-data/fine_data_flow/<dataset>_enhance1/0.8/test_500_dataset.tsv     --loop_num 12     --eval_loop_nums 2,4,6,8,12
```

注意：`finetune_looped.py` 末尾的 `__main__` 中硬编码了批量实验的数据集列表
（`file_list`）、预训练模型列表（`model_list`）与循环深度列表（`loop_num_list`），
批量跑实验时直接修改该处。

## 常用参数（finetune_looped.py）

| 参数 | 含义 |
| --- | --- |
| `--loop_num` | 循环次数上限（缺省读配置文件，再无则等于 layers_num） |
| `--dynamic_halting` | 动态早停（默认 True，自动切换 halting_looped_transformer 编码器） |
| `--halting_threshold` | 累计把握度停机阈值（默认 0.99） |
| `--ponder_weight` | 停机步数惩罚权重（默认 0.01） |
| `--halting_pooling` | 停机打分 pooling：mean / max / first / last |
| `--eval_loop_nums` | 逗号分隔的深度列表，测试时逐个深度评估 |
| `--only_test` | 仅测试 |
| `--labels_num` | 标签数（缺省从训练集统计；only_test 必填） |
| `--earlystop` | early stop 轮数（默认 5） |
