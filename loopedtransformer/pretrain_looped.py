# -*- coding: utf-8 -*-
"""循环 Transformer 流量预训练入口（参数共享基线）。

用法与 traffictransformer/pre-training/pretrain.py 基本一致，
差异只有两点：
1. 默认编码器为 looped_transformer，循环次数由 --loop_num 或配置文件控制。
2. 模型组装走 loopedtransformer.build.build_looped_model。

示例（在任意目录执行均可）：
    python3 loopedtransformer/pretrain_looped.py \
        --dataset_path traffictransformer/pre-training/dataset/dataset.pt \
        --vocab_path traffictransformer/vocab_gen/vocab_packet_4chars_all.txt \
        --config_path loopedtransformer/configs/looped_base_config.json \
        --target bertpackets \
        --loop_num 12 \
        --total_steps 120000

    # 随机深度 + 逐循环监督 + 一致性蒸馏（修复版预训练）
    python3 loopedtransformer/pretrain_looped.py \
        --dataset_path traffictransformer/pre-training/dataset/dataset.pt \
        --vocab_path traffictransformer/vocab_gen/vocab_packet_4chars_all.txt \
        --config_path loopedtransformer/configs/looped_random_sup_config.json \
        --target bertpackets \
        --total_steps 120000
"""
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# 路径自定位，不依赖固定的用户目录。
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_TRAFFIC_DIR = os.path.join(_REPO_ROOT, "traffictransformer")
for _p in (_REPO_ROOT, _TRAFFIC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import torch
import uer.trainer as trainer
from uer.utils.config import load_hyperparam
from uer.opts import model_opts, optimization_opts

from loopedtransformer.build import build_looped_model


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--log_file", type=str,
                        default=os.path.join(_TRAFFIC_DIR, "pre-training/train_looped_sup_no_random.log"))
    # Path options.
    parser.add_argument("--dataset_path", type=str,
                        default=os.path.join(_TRAFFIC_DIR, "pre-training/dataset/dataset.pt"),
                        help="Path of the preprocessed dataset.")
    parser.add_argument("--vocab_path", type=str,
                        default=os.path.join(_TRAFFIC_DIR, "vocab_gen/vocab_packet_4chars_all.txt"),
                        help="Path of the vocabulary file.")
    parser.add_argument("--spm_model_path", default=None, type=str,
                        help="Path of the sentence piece model.")
    parser.add_argument("--tgt_vocab_path", default=None, type=str,
                        help="Path of the target vocabulary file.")
    parser.add_argument("--tgt_spm_model_path", default=None, type=str,
                        help="Path of the target sentence piece model.")
    parser.add_argument("--pretrained_model_path", type=str, default=None,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_model_path", type=str,
                        default=os.path.join(_TRAFFIC_DIR,
                                             "pre-training/pre-training-model/looped_model_sup_no_random.bin"),
                        help="Path of the output model.")
    parser.add_argument("--config_path", type=str,
                        default=os.path.join(_THIS_DIR, "configs/looped_sup_no_random_config.json"),
                        help="Config file of model hyper-parameters.")

    # Training and saving options.
    parser.add_argument("--total_steps", type=int, default=120000, help="Total training steps.")
    parser.add_argument("--save_checkpoint_steps", type=int, default=10000,
                        help="Specific steps to save model checkpoint.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print prompt.")
    parser.add_argument("--accumulation_steps", type=int, default=1,
                        help="Specific steps to accumulate gradient.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size.")
    parser.add_argument("--instances_buffer_size", type=int, default=25600,
                        help="The buffer size of instances in memory.")
    parser.add_argument("--labels_num", type=int, required=False,
                        help="Number of prediction labels.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout value.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")

    # Preprocess options.
    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer.")

    # Model options.
    model_opts(parser)
    # 在原有 encoder 候选之外追加循环编码器，不修改 opts.py。
    for action in parser._actions:
        if action.dest == "encoder":
            action.choices = list(action.choices) + ["looped_transformer"]
            action.default = "looped_transformer"
    parser.add_argument("--loop_num", type=int, default=None,
                        help="循环次数。缺省时读取配置文件中的 loop_num，"
                             "配置文件也没有时等于 layers_num。")
    # 随机深度预训练与逐循环监督，缺省关闭，保持基线行为不变。
    parser.add_argument("--loop_num_min", type=int, default=None,
                        help="随机深度预训练的循环次数下界。设置后训练阶段每个 step 从 "
                             "[loop_num_min, loop_num] 均匀采样实际循环次数，"
                             "使任意深度的输出都被训练到。缺省表示固定 loop_num。")
    parser.add_argument("--loop_sup_weight", type=float, default=None,
                        help="逐循环监督损失权重。中间各轮循环的隐藏态接入共享 MLM/FM 头，"
                             "损失取平均后按该权重计入总损失，0 表示关闭。")
    parser.add_argument("--loop_distill_weight", type=float, default=None,
                        help="循环一致性蒸馏权重。中间各轮隐藏态向最后一轮（阻断梯度）"
                             "做 MSE 对齐，0 表示关闭。")
    parser.add_argument("--tgt_embedding",
                        choices=["word", "word_pos", "word_pos_seg", "word_sinusoidalpos"],
                        default="word_pos_seg", help="Target embedding type.")
    parser.add_argument("--decoder", choices=["transformer"], default="transformer",
                        help="Decoder type.")
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")
    parser.add_argument("--target",
                        choices=["bert", "bertflow", "bertpackets", "lm", "mlm", "bilm",
                                 "albert", "seq2seq", "t5", "cls", "prefixlm"],
                        default="bertpackets",
                        help="The training target of the pretraining model.")
    parser.add_argument("--tie_weights", action="store_true",
                        help="Tie the word embedding and softmax weights.")
    parser.add_argument("--has_lmtarget_bias", action="store_true",
                        help="Add bias on output_layer for lm target.")

    # MOE Model Options（基线不支持，仅为参数兼容保留）。
    parser.add_argument("--is_moe", action="store_true", help="adopt moe layer.")
    parser.add_argument("--vocab_size", type=int, required=False, help="Number of vocab.")
    parser.add_argument("--moebert_expert_dim", type=int, required=False, default=3072,
                        help="Dim of expert,default is ffn.")
    parser.add_argument("--moebert_expert_num", type=int, required=False, help="Number of expert.")
    parser.add_argument("--moebert_route_method",
                        choices=["gate-token", "gate-sentence", "hash-random", "hash-balance", "proto"],
                        default="hash-random", help="moebert route method.")
    parser.add_argument("--moebert_route_hash_list", default=None, type=str,
                        help="Path of moebert hash list file.")
    parser.add_argument("--moebert_load_balance", type=float, default=0.0,
                        help="gate loss weight.")

    # Masking options.
    parser.add_argument("--whole_word_masking", action="store_true", help="Whole word masking.")
    parser.add_argument("--span_masking", action="store_true", help="Span masking.")
    parser.add_argument("--span_geo_prob", type=float, default=0.2,
                        help="Hyperparameter of geometric distribution for span masking.")
    parser.add_argument("--span_max_length", type=int, default=10,
                        help="Max length for span masking.")

    # Optimizer options.
    optimization_opts(parser)

    # GPU options.
    parser.add_argument("--world_size", type=int, default=1,
                        help="Total number of processes (GPUs) for training.")
    parser.add_argument("--gpu_ranks", default=[0], nargs='+', type=int,
                        help="List of ranks of each process.")
    parser.add_argument("--master_ip", default="tcp://localhost:12345", type=str,
                        help="IP-Port of master for training.")
    parser.add_argument("--backend", choices=["nccl", "gloo"], default="nccl", type=str,
                        help="Distributed backend.")

    args = parser.parse_args()

    if args.config_path:
        args = load_hyperparam(args)
    # 命令行的 --loop_num 优先级高于配置文件。
    if args.loop_num is None:
        args.loop_num = getattr(args, "layers_num", 12)
    # 还原命令行显式指定的 loop_num（load_hyperparam 可能覆盖命名空间）。
    # 简化处理：配置文件中的 loop_num 可被 --loop_num 覆盖。
    cli_loop_num = None
    for i, a in enumerate(sys.argv):
        if a == "--loop_num" and i + 1 < len(sys.argv):
            cli_loop_num = int(sys.argv[i + 1])
    if cli_loop_num is not None:
        args.loop_num = cli_loop_num
    # 同理还原命令行显式指定的 loop_num_min。
    cli_loop_num_min = None
    for i, a in enumerate(sys.argv):
        if a == "--loop_num_min" and i + 1 < len(sys.argv):
            cli_loop_num_min = int(sys.argv[i + 1])
    if cli_loop_num_min is not None:
        args.loop_num_min = cli_loop_num_min

    # 三个新参数的归一化：配置文件与命令行都没给时取缺省值。
    args.loop_num_min = getattr(args, "loop_num_min", None)
    args.loop_sup_weight = getattr(args, "loop_sup_weight", None) or 0.0
    args.loop_distill_weight = getattr(args, "loop_distill_weight", None) or 0.0
    if args.loop_num_min is not None:
        assert 1 <= args.loop_num_min <= args.loop_num, \
            "loop_num_min 必须位于 [1, loop_num]。"

    ranks_num = len(args.gpu_ranks)

    if args.world_size > 1:
        assert torch.cuda.is_available(), "No available GPUs."
        assert ranks_num <= args.world_size, "Started processes exceed `world_size` upper limit."
        assert ranks_num <= torch.cuda.device_count(), "Started processes exceeds the available GPUs."
        args.dist_train = True
        args.ranks_num = ranks_num
        print("Using distributed mode for training.")
    elif args.world_size == 1 and ranks_num == 1:
        assert torch.cuda.is_available(), "No available GPUs."
        args.gpu_id = args.gpu_ranks[0]
        assert args.gpu_id < torch.cuda.device_count(), "Invalid specified GPU device."
        args.dist_train = False
        args.single_gpu = True
        print("Using GPU %d for training." % args.gpu_id)
    else:
        assert ranks_num == 0, "GPUs are specified, please check the arguments."
        args.dist_train = False
        args.single_gpu = False
        print("Using CPU mode for training.")

    # 关键注入点：把模型构建函数替换为循环版本，trainer 其余逻辑完全复用。
    trainer.build_model = build_looped_model
    # 开启逐循环监督或一致性蒸馏时，把 bertpackets 训练器替换为
    # 能读取 model.loop_aux_loss 的版本。两者都关闭时保持原训练器，
    # 行为与基线逐字节一致。
    if args.target == "bertpackets" and \
            (args.loop_sup_weight > 0.0 or args.loop_distill_weight > 0.0):
        from uer.trainer import str2trainer
        from loopedtransformer.looped_trainer import LoopedBertPacketsTrainer
        str2trainer["bertpackets"] = LoopedBertPacketsTrainer
    print(f"LoopedTransformer pretrain: encoder={args.encoder}, "
          f"loop_num={args.loop_num}, loop_num_min={args.loop_num_min}, "
          f"loop_sup_weight={args.loop_sup_weight}, "
          f"loop_distill_weight={args.loop_distill_weight}")
    def _save_model_without_mlm(model, model_path):
        if hasattr(model, "module"):
            model = model.module
        state_dict = {k: v for k, v in model.state_dict().items()
                    if not k.startswith(("target.mlm1.", "target.mlm2."))}
        torch.save(state_dict, model_path)
    '''
    trainer.save_model = _save_model_without_mlm 时，
    你是在修改 trainer 模块的 __dict__，
    把 save_model 这个名字重新指向了新函数。
    下次 train() 执行到 save_model(model, ...) 这一行，解析到的就是你替换后的函数。
    '''
    trainer.save_model = _save_model_without_mlm
    trainer.train_and_validate(args)


if __name__ == "__main__":
    main()
