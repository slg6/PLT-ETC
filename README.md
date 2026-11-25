# PLT-ETC

Encrypted traffic classification built on a **Looped Transformer** encoder. On top of a UER-style
pre-training framework, the encoder is replaced with a **weight-shared looped Transformer** that
supports **dynamic halting (Adaptive Computation Time, ACT)**: easy samples halt after one or two
loop iterations while hard ones iterate more, and `loop_num` only serves as a maximum-step
fallback — a single set of weights therefore covers multiple compute budgets.

## Repository layout

```
traffictransformer/        # Baseline system (data, vocab, pre-training, fine-tuning, uer framework)
├── fine-data/             # Fine-tuning data generation (pcap -> flow -> tsv)
├── vocab_gen/             # Traffic feature extraction and vocabulary generation
├── pre-training/          # MLM pre-training (preprocess.py -> pretrain.py)
├── fine-tuning/           # Baseline classification fine-tuning (run_classifier.py)
├── models/bert/           # BERT-family configuration files
├── uer/                   # Pre-training framework source code
└── 运行流程.md             # Original pipeline of the baseline system

loopedtransformer/         # Looped-Transformer additions (core contribution of this repo)
├── build.py               # Injects the looped encoders into the uer registry at runtime
├── looped_encoder.py      # Weight-shared looped encoder (baseline)
├── halting_looped_encoder.py  # Looped encoder with dynamic halting (halting_looped_transformer)
├── looped_model.py        # Model wrapper for the looped encoder (per-loop supervision + consistency distillation)
├── looped_trainer.py      # Trainer for looped pre-training
├── pretrain_looped.py     # Entry point for looped pre-training
├── finetune_looped.py     # Entry point for classification fine-tuning
└── configs/               # Hyper-parameter configs for the looped model (e.g. looped_random_sup_config.json)
```

## Key mechanisms

1. **Weight-shared looping**: a single Transformer layer is executed `loop_num` times.
2. **Dynamic halting (enabled by default, `--dynamic_halting`)**: per-sample cumulative confidence
   halts once it exceeds `--halting_threshold` (default 0.99); `--ponder_weight` (default 0.01)
   penalizes the number of halting steps so the model learns to stop early;
   `--halting_pooling` selects the pooling used for halting scores (follows `--pooling` by default).
   The halting logic follows the official Universal Transformer implementation
   (still_running / new_halted / remainders).
3. **Depth sweep evaluation**: `--eval_loop_nums 2,4,6,8,12` evaluates one weight checkpoint at
   every loop depth at test time and prints an accuracy-vs-depth table.
4. **Per-loop supervision + consistency distillation**: implemented in `looped_model.py` and
   `looped_trainer.py`, controlled by `loop_sup_weight` / `loop_distill_weight` (see configs).

> Note: the early experimental elastic-depth (`elastic_loop`) scheme is deprecated and has been
> removed from the codebase.

## Installation

```bash
pip install -r requirements.txt
```

A GPU environment is required (torch + CUDA).

## Pipeline

### 1. Data preparation and vocabulary (see traffictransformer/运行流程.md)

```bash
# pcap -> flow (use fine-data/SplitCap.exe on Windows)
python3 traffictransformer/fine-data/get_finedata.py
# traffic feature extraction -> vocabulary -> pre-training dataset
python3 traffictransformer/vocab_gen/extractIpTcp.py
python3 traffictransformer/vocab_gen/vocab_gen_all.py
python3 traffictransformer/pre-training/preprocess.py
```

### 2. Looped pre-training

```bash
python3 loopedtransformer/pretrain_looped.py     --dataset_path traffictransformer/pre-training/dataset/dataset.pt     --vocab_path traffictransformer/vocab_gen/vocab_packet_4chars_all.txt     --config_path loopedtransformer/configs/looped_random_sup_config.json     --target bertpackets     --loop_num 12     --total_steps 120000
```

### 3. Classification fine-tuning

```bash
python3 loopedtransformer/finetune_looped.py     --train_path traffictransformer/fine-data/fine_data_flow/<dataset>_enhance1/0.8/train_500_dataset.tsv     --dev_path   traffictransformer/fine-data/fine_data_flow/<dataset>_enhance1/0.8/valid_500_dataset.tsv     --test_path  traffictransformer/fine-data/fine_data_flow/<dataset>_enhance1/0.8/test_500_dataset.tsv     --loop_num 12     --eval_loop_nums 2,4,6,8,12
```

Note: the `__main__` block at the bottom of `finetune_looped.py` hard-codes the batch-experiment
lists (`file_list`, `model_list`, `loop_num_list`); edit that block when running batch experiments.

## Common arguments (finetune_looped.py)

| Argument | Description |
| --- | --- |
| `--loop_num` | Maximum loop count (falls back to the config file, then to `layers_num`) |
| `--dynamic_halting` | Dynamic halting (default True; switches to the halting_looped_transformer encoder) |
| `--halting_threshold` | Cumulative confidence threshold for halting (default 0.99) |
| `--ponder_weight` | Penalty weight on halting steps (default 0.01) |
| `--halting_pooling` | Pooling for halting scores: mean / max / first / last |
| `--eval_loop_nums` | Comma-separated depths, each evaluated at test time |
| `--only_test` | Test only |
| `--labels_num` | Number of labels (inferred from the training set if omitted; required with `--only_test`) |
| `--earlystop` | Early-stop patience in epochs (default 5) |
