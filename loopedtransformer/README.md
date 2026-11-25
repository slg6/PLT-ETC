# loopedtransformer

循环 Transformer 流量预训练模块。本目录与 `traffictransformer/` 平级，遵循《AI协作规则》，不修改 `traffictransformer/` 下的任何原始文件，通过继承与运行时外部注册复用原有组件。

## 当前进度

已完成思路分析中的第一步，参数共享循环编码器基线。整个编码器只含一个共享 Transformer 块，循环调用 `loop_num` 次，等效深度与原 12 层模型可比，参数量约为原来的 1/12。

在此之上新增两个预训练侧修复，解决"中间轮隐藏态从未被监督"导致的动态停机错配：随机深度（每 step 采样循环次数）与逐循环监督 + 一致性蒸馏。默认关闭，通过 `configs/looped_random_sup_config.json` 开启，不引入新参数。

## 目录结构

- `looped_encoder.py`：循环编码器，继承原始 `TransformerEncoder`，支持逐循环隐藏态输出以供表征诊断，支持随机深度预训练（训练时每 step 采样循环次数）。
- `looped_model.py`：模型外壳，替代原始 `Model.forward` 中按索引逐层调用的写法；内含逐循环监督与一致性蒸馏两个预训练机制（默认关闭）。
- `looped_trainer.py`：叠加循环附加损失的 bertpackets 训练器，日志中单列 `loop_aux` 一项。
- `build.py`：组件注册与模型组装。`register_looped_components` 在运行时把 `looped_transformer` 注入 `str2encoder`，不改动 uer 注册表文件。
- `pretrain_looped.py`：预训练入口，参数与 `traffictransformer/pre-training/pretrain.py` 一致，新增 `--loop_num`、`--loop_num_min`、`--loop_sup_weight`、`--loop_distill_weight`。
- `configs/looped_base_config.json`：基线配置。其中 `layers_num` 仅作兼容保留，实际深度由 `loop_num` 决定。
- `configs/looped_random_sup_config.json`：随机深度 + 逐循环监督配置。`loop_num_min=2` 开启随机深度，`loop_sup_weight=0.5`、`loop_distill_weight=0.1` 开启逐循环监督与一致性蒸馏。
- `tests/smoke_test.py`：冒烟测试，验证参数量压缩、前向反向与诊断接口。
- `tests/smoke_loop_pretrain.py`：冒烟测试，验证随机深度采样范围、附加损失可反向、不引入新参数。

## 预训练用法

```bash
python3 loopedtransformer/pretrain_looped.py \
    --dataset_path traffictransformer/pre-training/dataset/dataset.pt \
    --vocab_path traffictransformer/vocab_gen/vocab_packet_4chars_all.txt \
    --config_path loopedtransformer/configs/looped_base_config.json \
    --target bertpackets \
    --loop_num 12 \
    --total_steps 120000
```

数据预处理与词表生成沿用 `traffictransformer/运行流程.md` 中的原始步骤，无需改动。

## 随机深度与逐循环监督预训练

针对"微调阶段动态停机拿到的中间轮隐藏态从未被监督"的错配问题，预训练侧提供两个可叠加的机制，均默认关闭、不引入新参数：

- `--loop_num_min 2`：随机深度。训练模式每个 step 从 `[loop_num_min, loop_num]` 均匀采样实际循环次数，使任意深度的输出都被训练到；评估模式始终跑满 `loop_num`。
- `--loop_sup_weight 0.5`：逐循环监督。中间各轮隐藏态接入共享的 MLM/FM 头，损失取平均后按权重计入总损失。
- `--loop_distill_weight 0.1`：循环一致性蒸馏。中间各轮隐藏态向最后一轮（阻断梯度）做 MSE 对齐，防止表征漂移。

两项附加损失由 `LoopedModel` 暂存在 `loop_aux_loss`，经 `LoopedBertPacketsTrainer` 叠加进总损失，日志单列 `loop_aux`；`loss_mlm/loss_sp/loss_fm` 口径与基线完全一致。使用 `configs/looped_random_sup_config.json` 即可一次开启三项。产出的 checkpoint 键名与基线对齐，微调入口无需任何改动。

```bash
python3 loopedtransformer/pretrain_looped.py \
    --dataset_path traffictransformer/pre-training/dataset/dataset.pt \
    --vocab_path traffictransformer/vocab_gen/vocab_packet_4chars_all.txt \
    --config_path loopedtransformer/configs/looped_random_sup_config.json \
    --target bertpackets \
    --total_steps 120000
```

## 微调用法

```bash
python3 loopedtransformer/finetune_looped.py \
    --pretrained_model_path traffictransformer/pre-training/pre-training-model/looped_model.bin-120000 \
    --train_path traffictransformer/fine-data/train.tsv \
    --dev_path traffictransformer/fine-data/dev.tsv \
    --test_path traffictransformer/fine-data/test.tsv \
    --config_path loopedtransformer/configs/looped_base_config.json \
    --loop_num 12 \
    --epochs_num 10
```

循环预训练权重只含一个共享块，与循环分类器的键名天然对齐，直接加载即可，无需权重广播。微调数据准备沿用 `traffictransformer/运行流程.md` 中的原始步骤。

两个循环特有的功能：

- `--elastic_loop --loop_num_min 2 --loop_num_max 12`：弹性深度微调。每个训练步随机采样循环次数，使一套权重适应多种深度。
- `--eval_loop_nums 2,4,6,8,12`：测试阶段逐个深度评估，输出精度-深度对照表，用于展示一套权重覆盖多档算力预算的特性。
- `--dynamic_halting [--halting_threshold 0.99] [--ponder_weight 0.01]`：借鉴 Universal Transformer 的 ACT 动态早停。停机头（hidden->1 的 sigmoid 打分器）逐轮给每个样本打把握度，累计超过阈值即冻结该样本的隐藏态，`loop_num` 仅作 max_steps 兜底。ponder penalty 把平均停机步数加进损失，逼模型自己权衡精度与算力。评估时打印 `Avg. halting steps`，与 `loop_num` 对比即可观察早停效果；评估模式下整个 batch 全部停机还会提前退出循环。与 `--elastic_loop` 互斥。该模式自动切换 `halting_looped_transformer` 编码器，停机头不参与预训练，从预训练权重加载时随机初始化，微调阶段随分类损失端到端学习。

## 测试

```bash
python3 loopedtransformer/tests/smoke_test.py
python3 loopedtransformer/tests/smoke_finetune.py
python3 loopedtransformer/tests/smoke_halting.py
python3 loopedtransformer/tests/smoke_loop_pretrain.py
```

## 已知边界

1. 暂不支持 MoE 与非 fully_visible 掩码，调用时会直接报错。
2. 微调仅支持单卡或 CPU。循环预训练权重加载进原始 12 层结构（非循环编码器）仍需权重广播，本目录的微调入口不涉及该路径。
3. 与原始模型的对比实验必须区分两组口径。iso-param 对比 12 层非共享模型，iso-FLOP 对比 144 层量级的非共享模型。
