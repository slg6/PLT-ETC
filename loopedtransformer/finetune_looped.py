# -*- coding: utf-8 -*-


import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_TRAFFIC_DIR = os.path.join(_REPO_ROOT, "traffictransformer")
for _p in (_REPO_ROOT, _TRAFFIC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


import argparse
import random

import torch
import torch.nn as nn
import tqdm
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

from uer.encoders import str2encoder
from uer.layers import str2embedding
from uer.model_saver import save_model
from uer.opts import finetune_opts
from uer.utils import str2optimizer, str2scheduler, str2tokenizer
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed

from loopedtransformer.build import register_looped_components


class LoopedClassifier(nn.Module):

    def __init__(self, args):
        super(LoopedClassifier, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)
        self.ponder_weight = getattr(args, "ponder_weight", 0.0)
        self.act_stats = None

    def _encode(self, src, seg):
        emb = self.embedding(src, seg)
        output = self.encoder(emb, seg)
        self.act_stats = None
        if isinstance(output, (tuple, list)):
            if len(output) >= 2 and isinstance(output[1], dict):
                self.act_stats = output[1]
            output = output[0]
        return output

    def forward(self, src, tgt, seg):
        output = self._encode(src, seg)
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        if tgt is not None:
            loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            if self.act_stats is not None and self.ponder_weight > 0.0:
                loss = loss + self.ponder_weight * self.act_stats["n_updates"].mean()
            return loss, logits
        return None, logits


def count_labels_num(path):
    labels_set = set()
    with open(path, mode="r", encoding="utf-8") as f:
        columns = {}
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            labels_set.add(int(line.strip().split("\t")[columns["label"]]))
    return len(labels_set)


def read_dataset(args, path):
    dataset, columns = [], {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line[:-1].split("\t")
            tgt = int(line[columns["label"]])
            text = line[columns["text_a"]].split()
            src = args.tokenizer.convert_tokens_to_ids(text)
            if len(src) < 1 or (len(src) - 1) % 11 != 0:
                raise ValueError(
                    "第 %d 行：text_a 含 %d 个 token，不符合 "
                    "1 个 [CLS] + 每包 20 token 的格式。"
                    % (line_id + 1, len(src)))
            seg = [1]
            i = 1
            # guard = 0
            while True:
                ip_seg = [i] * 7
                i += 1
                seg.extend(ip_seg)
                tcp_seg = [i] * 4
                i += 1
                seg.extend(tcp_seg)
                if len(seg) == len(src):
                    break
                # guard += 1
                # if guard > 10000:
                #     raise ValueError(
            if len(src) > args.seq_length:
                src = src[: args.seq_length]
                seg = seg[: args.seq_length]
            while len(src) < args.seq_length:
                src.append(0)
                seg.append(0)
            dataset.append((src, tgt, seg))
    return dataset


def batch_loader(batch_size, src, tgt, seg):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        yield (src[i * batch_size: (i + 1) * batch_size, :],
               tgt[i * batch_size: (i + 1) * batch_size],
               seg[i * batch_size: (i + 1) * batch_size, :])
    if instances_num > instances_num // batch_size * batch_size:
        yield (src[instances_num // batch_size * batch_size:, :],
               tgt[instances_num // batch_size * batch_size:],
               seg[instances_num // batch_size * batch_size:, :])


def build_optimizer(args, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay_rate": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay_rate": 0.0},
    ]
    if args.optimizer in ["adamw"]:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters,
                                                  lr=args.learning_rate, correct_bias=False)
    else:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters,
                                                  lr=args.learning_rate,
                                                  scale_parameter=False, relative_step=False)
    if args.scheduler in ["constant"]:
        scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps * args.warmup)
    else:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps * args.warmup,
                                                  args.train_steps)
    return optimizer, scheduler


def evaluate(args, model, dataset, print_confusion_matrix=False):
    src = torch.LongTensor([sample[0] for sample in dataset])
    tgt = torch.LongTensor([sample[1] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])

    correct = 0
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)
    y_true, y_pred = [], []
    ponder_sum, ponder_cnt = 0.0, 0
    model.eval()
    for src_batch, tgt_batch, seg_batch in batch_loader(args.batch_size, src, tgt, seg):
        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        with torch.no_grad():
            _, logits = model(src_batch, tgt_batch, seg_batch)
        stats = getattr(model, "act_stats", None)
        if stats is not None:
            ponder_sum += stats["n_updates"].sum().item()
            ponder_cnt += stats["n_updates"].numel()
        pred = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
        for j in range(pred.size()[0]):
            confusion[pred[j], tgt_batch[j]] += 1
        y_true.extend(tgt_batch.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
        correct += torch.sum(pred == tgt_batch).item()

    metrics = {
        "acc": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "macro_p": precision_score(y_true, y_pred, average="macro"),
        "macro_r": recall_score(y_true, y_pred, average="macro"),
        "avg_halting": (ponder_sum / ponder_cnt) if ponder_cnt > 0 else None,
    }
    print("Acc. (Correct/Total): {:.4f} ({}/{})".format(
        metrics["acc"], correct, len(dataset)))
    print("Macro precision: {:.4f}, Macro recall: {:.4f}, Macro f1: {:.4f}".format(
        metrics["macro_p"], metrics["macro_r"], metrics["macro_f1"]))
    if ponder_cnt > 0:
        print("Avg. halting steps: {:.2f} / {}".format(
            ponder_sum / ponder_cnt,
            getattr(model.encoder, "loop_num", "?")))
    if print_confusion_matrix:
        print("Confusion matrix:")
        torch.set_printoptions(threshold=1000000, edgeitems=1000000, linewidth=1000000)
        print(confusion)
    return metrics


def load_pretrained_parameters(args, model):
    if args.pretrained_model_path and os.path.exists(args.pretrained_model_path):
        print("Initialize with pretrained model. %s" % args.pretrained_model_path)
        state = torch.load(args.pretrained_model_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("从预训练权重加载 %d 项，跳过预训练独有的 %d 项（target 头等），"
              "随机初始化 %d 项（分类头与停机头）。"
              % (len(state) - len(unexpected), len(unexpected), len(missing)))
    else:
        print("Initialize with normal distribution.")
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)


def eval_loop_sweep(args, model, testset):
    if not hasattr(model.encoder, "loop_num"):
        print("当前编码器不支持调节循环深度，跳过深度扫描。")
        return
    depths = [int(x) for x in args.eval_loop_nums.split(",") if x.strip()]
    if not depths:
        return
    original = model.encoder.loop_num
    print("\n循环深度扫描评估（同一套权重，仅改变循环次数）：")
    print("%10s %10s %10s" % ("loop_num", "acc", "macro_f1"))
    for n in depths:
        model.encoder.loop_num = n
        m = evaluate(args, model, testset)
        print("%10d %10.4f %10.4f" % (n, m["acc"], m["macro_f1"]))
    model.encoder.loop_num = original


def main(dataset='ustc',pretrained_model='looped_model_sup.bin-120000', loop_num = 12):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    finetune_opts(parser)
    for action in parser._actions:
        if action.dest == "encoder":
            action.choices = list(action.choices) + ["looped_transformer",
                                                     "halting_looped_transformer"]
            action.default = "looped_transformer"
        if action.dest == "pretrained_model_path":
            action.default = "traffictransformer/pre-training/pre-training-model/looped_model.bin-120000"
        if action.dest == "output_model_path":
            action.default = os.path.join(_THIS_DIR, f"finetune-output/{dataset}/best_model.bin")
        if action.dest == "config_path":
            action.default = os.path.join(_THIS_DIR, "configs/looped_base_config.json")
        if action.dest == "vocab_path":
            action.default = os.path.join(
                _TRAFFIC_DIR, "vocab_gen/vocab_packet_4chars_all.txt")
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"],
                        default="first", help="Pooling type.")
    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer.")
    parser.add_argument("--labels_num", type=int, default=None,
                        help="标签数。缺省时从训练集统计，only_test 模式下必填。")
    parser.add_argument("--earlystop", type=int, default=5, help="early stop rounds.")
    parser.add_argument("--only_test", action="store_true", help="Only run test.")
    parser.add_argument("--loop_num", type=int, default=None,
                        help="循环次数。缺省时读取配置文件，配置文件也没有时等于 layers_num。")
    parser.add_argument("--eval_loop_nums", type=str, default="",
                        help='逗号分隔的深度列表，如 "2,4,6,8,12"。'
                             "测试阶段逐个深度评估，输出精度-深度对照表。")
    parser.add_argument("--dynamic_halting", type=bool, default=True,
                        help="动态早停：逐样本累计把握度达标即停机，"
                             "自动切换 halting_looped_transformer 编码器。")
    parser.add_argument("--halting_threshold", type=float, default=0.99,
                        help="累计把握度停机阈值，略小于 1。")
    parser.add_argument("--ponder_weight", type=float, default=0.01,
                        help="停机步数惩罚权重，越大模型越早停。")
    parser.add_argument("--halting_pooling", choices=["mean", "max", "first", "last"],
                        default=None, help="停机打分用的 pooling，缺省跟随 --pooling。")
    parser.add_argument("--is_moe", action="store_true", help="adopt moe layer.")
    parser.add_argument("--moebert_expert_dim", type=int, default=3072)
    parser.add_argument("--moebert_expert_num", type=int, default=None)
    parser.add_argument("--moebert_route_method",
                        choices=["gate-token", "gate-sentence", "hash-random",
                                 "hash-balance", "proto"], default="hash-random")
    parser.add_argument("--moebert_route_hash_list", default=None, type=str)
    parser.add_argument("--moebert_load_balance", type=float, default=0.0)

    args = parser.parse_args()
    # args.pretrained_model_path = f"traffictransformer/pre-training/pre-training-model/looped_model_sup.bin-120000"
    args.pretrained_model_path = f"traffictransformer/pre-training/pre-training-model/{pretrained_model}"
    args.output_model_path = f"loopedtransformer/finetune-output-sup/{dataset}/{pretrained_model}"
    args.train_path = f"traffictransformer/fine-data/fine_data_flow/{dataset}_enhance1/0.8/train_500_dataset.tsv"
    args.dev_path = f"traffictransformer/fine-data/fine_data_flow/{dataset}_enhance1/0.8/valid_500_dataset.tsv"
    args.test_path = f"traffictransformer/fine-data/fine_data_flow/{dataset}_enhance1/0.8/test_500_dataset.tsv"
    args.epochs_num = 100
    args.batch_size = 32

    if args.config_path:
        args = load_hyperparam(args)
    cli_loop_num = None
    for i, a in enumerate(sys.argv):
        if a == "--loop_num" and i + 1 < len(sys.argv):
            cli_loop_num = int(sys.argv[i + 1])
    if cli_loop_num is not None:
        args.loop_num = cli_loop_num
    if args.loop_num is None:
        args.loop_num = getattr(args, "layers_num", 12)
    args.loop_num = loop_num
    if args.loop_num != 12:
        args.output_model_path = f"loopedtransformer/finetune-output-sup/{dataset}/loop_num_max{str(args.loop_num)}.bin"
    if pretrained_model=='looped_model_sup.bin-120000' and args.loop_num == 12:
        args.output_model_path = f"loopedtransformer/finetune-output-sup/{dataset}/best_model.bin"
    if args.dynamic_halting:
        args.encoder = "halting_looped_transformer"

    set_seed(args.seed)
    register_looped_components()

    if args.labels_num is None:
        assert args.train_path is not None, "only_test 模式下必须显式提供 --labels_num。"
        args.labels_num = count_labels_num(args.train_path)

    args.tokenizer = str2tokenizer[args.tokenizer](args)

    model = LoopedClassifier(args)
    if hasattr(model.encoder, "loop_num"):
        model.encoder.loop_num = args.loop_num

    load_pretrained_parameters(args, model)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)
    print("Finetune: encoder=%s, loop_num=%d"
          % (args.encoder, getattr(model.encoder, "loop_num", -1)))

    if args.only_test:
        assert args.test_path is not None, "only_test 模式需要提供 --test_path。"
        testset = read_dataset(args, args.test_path)
        evaluate(args, model, testset, print_confusion_matrix=True)
        eval_loop_sweep(args, model, testset)
        return

    trainset = read_dataset(args, args.train_path)
    random.shuffle(trainset)
    instances_num = len(trainset)
    src = torch.LongTensor([example[0] for example in trainset])
    tgt = torch.LongTensor([example[1] for example in trainset])
    seg = torch.LongTensor([example[2] for example in trainset])
    args.train_steps = int(instances_num * args.epochs_num / args.batch_size) + 1

    print("Batch size: ", args.batch_size)
    print("The number of training instances:", instances_num)

    optimizer, scheduler = build_optimizer(args, model)

    output_dir = os.path.dirname(os.path.abspath(args.output_model_path))
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "finetune_log.txt")
    if args.dynamic_halting is False:
        with open(log_path, "a") as f:
            f.write("强制模型跑满12轮次\n")
        args.output_model_path = f"loopedtransformer/finetune-output-sup/{dataset}/model_all12.bin"
    with open(log_path, "a") as f:
        if pretrained_model == "none":
            f.write("pretrained model: none\n")
        else:
            f.write("pretrained model: %s\n" % args.pretrained_model_path)
        f.write("loop_num_max: %d\n" % args.loop_num)

    total_loss, best_result, best_result_round = 0.0, 0.0, 0
    for epoch in tqdm.tqdm(range(1, args.epochs_num + 1)):
        model.train()
        for i, (src_batch, tgt_batch, seg_batch) in enumerate(
                batch_loader(args.batch_size, src, tgt, seg)):
            model.zero_grad()
            loss, _ = model(src_batch.to(args.device),
                            tgt_batch.to(args.device),
                            seg_batch.to(args.device))
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                lr = optimizer.param_groups[0]["lr"]
                print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}, lr: {:.3e}".format(
                    epoch, i + 1, total_loss / args.report_steps, lr))
                total_loss = 0.0

        lr = optimizer.param_groups[0]["lr"]
        if hasattr(model.encoder, "loop_num"):
            model.encoder.loop_num = args.loop_num
        result = evaluate(args, model, read_dataset(args, args.dev_path))
        halt = "N/A"
        if result["avg_halting"] is not None:
            halt = "%.2f/%d" % (result["avg_halting"],
                                getattr(model.encoder, "loop_num", 0))
        with open(log_path, "a") as f:
            f.write("epoch {:3d}, valid macro f1: {:.4f}, acc: {:.4f}, "
                    "avg halting: {}, lr: {:.4e}\n".format(
                        epoch, result["macro_f1"], result["acc"], halt, lr))
        if result["macro_f1"] > best_result:
            best_result = result["macro_f1"]
            best_result_round = epoch
            save_model(model, args.output_model_path)
        elif epoch - best_result_round >= args.earlystop:
            print("early stopping...")
            break

    if args.test_path is not None:
        print("Test set evaluation.")
        model.load_state_dict(torch.load(args.output_model_path, map_location=args.device))
        testset = read_dataset(args, args.test_path)
        result = evaluate(args, model, testset, print_confusion_matrix=True)
        halt = "N/A"
        if result["avg_halting"] is not None:
            halt = "%.2f/%d" % (result["avg_halting"],
                                getattr(model.encoder, "loop_num", 0))
        with open(log_path, "a") as f:
            f.write("test macro f1: {:.4f}, acc: {:.4f}, pr: {:.4f}, rc: {:.4f}, "
                    "avg halting: {}, best_epoch: {:4d}\n".format(
                        result["macro_f1"], result["acc"], result["macro_p"],
                        result["macro_r"], halt, best_result_round))
        eval_loop_sweep(args, model, testset)


if __name__ == "__main__":
    file_list = ["Edge-IIoTset", "ustc", "cstent", "cipherspectrum_mix",
                "cipherspectrum_aes256", "cipherspectrum_aes128", "cipherspectrum_cha", "itcnet"]
    # file_list = ['ustc']
    # file_list = ["itcnet"]
    file_list = ["Edge-IIoTset", "ustc", "itcnet"]
    model_list = ["looped_model_sup.bin-120000", "looped_model_sup_no_distillweight.bin-120000", "looped_model_sup_no_supdistill.bin-120000",
                   "looped_model_sup_no_supweight.bin-120000"]
    model_list = ["looped_model_sup.bin-120000"]
    loop_num_list = [3, 6, 9]
    loop_num_list = [24]
    # model_list = ["looped_model_sup_no_random.bin-120000"]
    # model_list = ["none"]
    for dataset in file_list:
        for pretrained_model in model_list:
            for loop_num in loop_num_list:
                print("=" * 60)
                print("dataset: %s" % (dataset))
                main(dataset=dataset, pretrained_model=pretrained_model, loop_num=loop_num)
