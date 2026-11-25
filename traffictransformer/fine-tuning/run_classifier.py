"""
This script provides an exmaple to wrap UER-py for classification.
"""
import os
import sys
# os.chdir(os.path.expanduser("~/llmfortraffic/looptransformer/traffictransformer"))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.getcwd())
import random
import argparse
import torch
import torch.nn as nn
from uer.layers import *
from uer.encoders import *
from uer.utils.constants import *
from uer.utils import *
from uer.utils.optimizers import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from uer.opts import finetune_opts
import tqdm
import numpy as np
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class Classifier(nn.Module):        #分类层就两个线性层，中间使用一个tanh激活函数进行连接。
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab)) #词嵌入层 tokenID --> token向量  每个token都被转换为了一个向量，这个向量包含了：语义信息、位置信息、段落信息
        self.encoder = str2encoder[args.encoder](args)  #编码器层 编码器层由多个Transformer层堆叠而成 生成最终表示向量  token向量 --> CLS



        self.labels_num = args.labels_num   #标签数量
        self.pooling = args.pooling     # 池化方式，默认"first"取CLS
        self.soft_targets = args.soft_targets   #软标签参数（知识蒸馏相关）  # 是否使用软标签
        self.soft_alpha = args.soft_alpha       # 软标签权重
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size) # 输出层参数
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)  # 输出层参数
        #预训练阶段，主要训练两个核心组件：嵌入层和编码器层。  微调阶段， 在预训练的基础上添加分类头。
        
    def forward(self, src, tgt, seg, soft_tgt=None):
        # Embedding.
        emb = self.embedding(src, seg)  #torch.Size([21, 320, 768])#完成了词嵌入、位置嵌入、段落嵌入。每个token都被转换为了一个向量，这个向量包含了：语义信息、位置信息、段落信息
        # Encoder.
        output = self.encoder(emb, seg)[0] #生成强大的序列表示，尤其是 [CLS] token 的表示可以用于下游任务
        # Target.
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]    # First.取第一个token的输出：即CLS
        output = torch.tanh(self.output_layer_1(output))    #CLS先经过了一个线性层和tanh激活函数
        logits = self.output_layer_2(output)        #在经过一个线性层得到输出结果
        if tgt is not None:
            if self.soft_targets and soft_tgt is not None:
                loss = self.soft_alpha * nn.MSELoss()(logits, soft_tgt) + \
                       (1 - self.soft_alpha) * nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            else:
                loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))        #计算损失函数
            return loss, logits     #返回损失值  预测结果
        else:
            return None, logits
            #return temp_output, logits
    #slg
    '''
    src: 输入序列的token ID
    seg: 用于区分不同序列片段
    tgt 的主要特点：
        形状为 [batch_size]
        每个元素是一个整数，表示对应样本的类别
        在训练时用于计算损失函数
        在评估时用作真实标签(ground truth)
    '''
    def get_cls_vector(self, src, seg):  #获取CLS向量。
        with torch.no_grad():
            emb = self.embedding(src, seg)
            output = self.encoder(emb, seg)
            cls_vector = output[:, 0, :]
        return cls_vector

    #slg

def count_labels_num(path):
    labels_set, columns = set(), {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line.strip().split("\t")
            label = int(line[columns["label"]])
            labels_set.add(label)
    return len(labels_set)


def load_or_initialize_parameters(args, model):
    if args.pretrained_model_path is not None and os.path.exists(args.pretrained_model_path):
        print(f"Initialize with pretrained model. {args.pretrained_model_path}")
        model.load_state_dict(torch.load(args.pretrained_model_path, map_location={'cuda:1':'cuda:0', 'cuda:2':'cuda:0', 'cuda:3':'cuda:0'}), strict=False)
    else:
        print("Initialize with normal distribution.")
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)


def build_optimizer(args, model):
    param_optimizer = list(model.named_parameters())    # 获取模型所有命名参数(name, parameter)元组列表
    no_decay = ['bias', 'gamma', 'beta']    # 定义不需要权重衰减的参数名列表 权重衰减：在每次参数更新时，将权重乘以一个小于1的因子。 为了包含归一化的参数
    optimizer_grouped_parameters = [         # 将参数分为两组，用于不同的权重衰减策略
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    if args.optimizer in ["adamw"]: # 根据指定的优化器类型创建优化器
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    else:       # 其他优化器：不使用参数缩放和相对步长
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                  scale_parameter=False, relative_step=False)
    if args.scheduler in ["constant"]:      # 根据指定的学习率调度策略创建调度器
        scheduler = str2scheduler[args.scheduler](optimizer)    # 常数学习率
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup)  # 带预热的常数学习率
    else:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup, args.train_steps)# 其他调度策略（如线性衰减）
    return optimizer, scheduler


def batch_loader(batch_size, src, tgt, seg, soft_tgt=None):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size : (i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size : (i + 1) * batch_size]
        seg_batch = seg[i * batch_size : (i + 1) * batch_size, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[i * batch_size : (i + 1) * batch_size, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size :, :]
        tgt_batch = tgt[instances_num // batch_size * batch_size :]
        seg_batch = seg[instances_num // batch_size * batch_size :, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[instances_num // batch_size * batch_size :, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None


def read_dataset(args, path):   #读取数据集并将其转换为模型可处理的格式。
    dataset, columns = [], {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line[:-1].split("\t")
            tgt = int(line[columns["label"]])
            if args.soft_targets and "logits" in columns.keys():
                soft_tgt = [float(value) for value in line[columns["logits"]].split(" ")]
            text = line[columns["text_a"]]
            text = [t for t in text.split()]
            src = args.tokenizer.convert_tokens_to_ids(text)    #转化为了input_ids
            seg = [0]
            i = 0
            while True:
                ip_seg = [i] * 11
                i += 1
                seg.extend(ip_seg)
                tcp_seg = [i] * 9
                i += 1
                seg.extend(tcp_seg)
                if len(seg) == len(src):
                    break
            if len(src) > args.seq_length:      #把超长向量截取成固定长度
                src = src[: args.seq_length]
                seg = seg[: args.seq_length]
            while len(src) < args.seq_length:   #把不够长的向量填充成固定长度
                src.append(0)
                seg.append(0)
            if args.soft_targets and "logits" in columns.keys():    #软标签：与蒸馏有关
                dataset.append((src, tgt, seg, soft_tgt))
            else:
                dataset.append((src, tgt, seg))     #token ID序列、标签、段落标识

    return dataset  

def train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch=None):
    model.zero_grad()

    src_batch = src_batch.to(args.device)
    tgt_batch = tgt_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)   #torch.Size([21, 320])
    if soft_tgt_batch is not None:
        soft_tgt_batch = soft_tgt_batch.to(args.device)

    loss, _ = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch)
    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss)

    # if args.fp16:
    #     with args.amp.scale_loss(loss, optimizer) as scaled_loss:
    #         scaled_loss.backward()
    # else:
    loss.backward()

    optimizer.step()    #根据梯度更新参数
    scheduler.step()    #根据调度策略调整学习率

    return loss #损失值


def evaluate(args, dataset, print_confusion_matrix=False):  #进行模型评估
    src = torch.LongTensor([sample[0] for sample in dataset])
    tgt = torch.LongTensor([sample[1] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])

    batch_size = args.batch_size

    correct = 0 # 正确预测的样本数
    # Confusion matrix.
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long) # 初始化混淆矩阵，大小为 labels_num x labels_num
    y_true, y_pred = [], [] # 存储真实标签和预测标签
    args.model.eval()    # 将模型设置为评估模式

    for i, (src_batch, tgt_batch, seg_batch, _) in enumerate(batch_loader(batch_size, src, tgt, seg)):
        src_batch = src_batch.to(args.device)
        # print(src_batch[0][113],args.tokenizer.convert_ids_to_tokens([src_batch.cpu().numpy()[0][113]]))
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        with torch.no_grad():   # 禁用梯度计算，进行前向传播
            _, logits = args.model(src_batch, tgt_batch, seg_batch)
        pred = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)   # 获取预测结果
        gold = tgt_batch        # 真实标签
        for j in range(pred.size()[0]):
            confusion[pred[j], gold[j]] += 1     # 更新混淆矩阵
            y_true.append(gold[j].cpu())     # 记录真实标签
            y_pred.append(pred[j].cpu())    # 记录预测标签
        correct += torch.sum(pred == gold).item()    # 统计正确预测的数量
        
    
    if print_confusion_matrix:  # 如果需要打印混淆矩阵
        print("Confusion matrix:")
        torch.set_printoptions(threshold=1000000, edgeitems=1000000, linewidth=1000000)
        print(confusion)
        cf_array = confusion.numpy()
        # with open("./results/confusion_matrix",'w') as f:
        #     for cf_a in cf_array:
        #         f.write(str(cf_a)+'\n')
        print("Report precision, recall, and f1:")  # 计算每个类别的精确率、召回率和F1值
        eps = 1e-9
        for i in range(confusion.size()[0]):
            p = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)
            r = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)
            if (p + r) == 0:
                f1 = 0
            else:
                f1 = 2 * p * r / (p + r)
            print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i, p, r, f1))
        
    # 打印总体评估指标
    print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct / len(dataset), correct, len(dataset)))    # 准确率
    print("Macro precision: {:.4f}, Micro precision: {:.4f}, Weighted precision: {:.4f}".format(        # 各种平均方式下的精确率
        precision_score(y_true,y_pred,average='macro'), precision_score(y_true,y_pred,average='micro'), precision_score(y_true,y_pred,average='weighted')))
    print("Macro recall: {:.4f}, Micro recall: {:.4f}, Weighted recall: {:.4f}".format(     # 各种平均方式下的召回率
        recall_score(y_true,y_pred,average='macro'), recall_score(y_true,y_pred,average='micro'), recall_score(y_true,y_pred,average='weighted')))
    print("Macro f1: {:.4f}, Micro f1: {:.4f}, Weighted f1: {:.4f}".format(     # 各种平均方式下的F1值
        f1_score(y_true,y_pred,average='macro'), f1_score(y_true,y_pred,average='micro'), f1_score(y_true,y_pred,average='weighted')))
    '''宏平均值：对每个类别分别计算F1值、将所有类别的F1值相加后取平均'''
    
    return f1_score(y_true,y_pred,average='macro'), accuracy_score(y_true,y_pred),precision_score(y_true,y_pred,average='macro'),recall_score(y_true,y_pred,average='macro'),confusion   # 返回宏平均F1值和混淆矩阵

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    finetune_opts(parser)

    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")
    
    parser.add_argument("--earlystop", type=int, default=5, help="early stop rounds.")
    parser.add_argument("--samples", type=int, default=0, help="samples.")
    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer."
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space."
                             )

    parser.add_argument("--soft_targets", action='store_true',
                        help="Train model with logits.")
    parser.add_argument("--soft_alpha", type=float, default=0.5,
                        help="Weight of the soft targets loss.")
    
    #MOE Model Options
    parser.add_argument("--is_moe", action="store_true", help="adopt moe layer.")
    parser.add_argument("--vocab_size", type=int, required=False, help="Number of vocab.")
    parser.add_argument("--moebert_expert_dim", type=int, required=False, default=3072, help="Dim of expert,default is ffn.")
    parser.add_argument("--moebert_expert_num", type=int, required=False, help="Number of expert.")
    parser.add_argument("--moebert_route_method", choices=["gate-token", "gate-sentence", "hash-random", "hash-balance","proto"], default="hash-random",
                        help="moebert route method.")
    parser.add_argument("--moebert_route_hash_list", default=None, type=str, help="Path of moebert hash list file.")
    parser.add_argument("--moebert_load_balance", type=float, default=0.0, help="gate loss weight.")
    # parser.add_argument('--only_test', type=bool, default=False, help="not train, just test.")
    parser.add_argument('--only_test', action='store_true', help='Only run test')

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)        #加载分类模型架构 ！！！    默认从models/bert/base_config.json中加载

    set_seed(args.seed)     #函数用于设置随机种子，确保实验的可重复性。args.seed默认为7

    # Count the number of labels.
    if args.train_path is None:
        args.labels_num = 197
    else:
        args.labels_num = count_labels_num(args.train_path) #得到label的数量

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)    #设置分词器

    # Build classification model.
    model = Classifier(args)    #微调后模型的结构。  在预训练模型后面加上了一个分类头。

    # Load or initialize parameters.
    load_or_initialize_parameters(args, model)  #加载模型

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)       #模型加载到设置的设备(GPU、CPU)上

    if args.train_path is None:
        args.model = model
        args.labels_num = 197
        print("No train data, only evaluate..")
        result = evaluate(args, read_dataset(args, args.dev_path))
        return
    
    # Training phase.
    trainset = read_dataset(args, args.train_path)  #token ID序列、标签、段标记 [([...], 1, [...]), ([...], 0, [...]), ...]
    random.shuffle(trainset)
    instances_num = len(trainset)   #样本数
    batch_size = args.batch_size   
    
    src = torch.LongTensor([example[0] for example in trainset])    #torch.Size([样本数, 截取字节数]) token is
    tgt = torch.LongTensor([example[1] for example in trainset])     #torch.Size([样本数])   label
    seg = torch.LongTensor([example[2] for example in trainset])     #torch.Size([样本数, 截取字节数])   段标记
    if args.soft_targets:
        soft_tgt = torch.FloatTensor([example[3] for example in trainset])
    else:
        soft_tgt = None

    # args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1    #计算总训练步数
    args.train_steps = int(instances_num * 100 / batch_size) + 1    #计算总训练步数

    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)

    optimizer, scheduler = build_optimizer(args, model) #定义了训练过程中的优化器、学习率的调整策略

    # if args.fp16:
    #     try:
    #         from apex import amp
    #     except ImportError:
    #         raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    #     model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    #     args.amp = amp

    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    args.model = model

    total_loss, result, best_result = 0.0, 0.0, 0.0
    best_result_round = 0
    # print("Start training.")
    loss_save_path = args.dev_path.rsplit('/', 1)[0]+"/nofeature_"+str(args.samples)+"_"+args.pretrained_model_path.rsplit('/', 1)[1].rsplit('-', 1)[1]
    if not args.only_test:
        for epoch in tqdm.tqdm(range(1, args.epochs_num + 1)):
            model.train()   # 设置模型为训练模式（启用dropout等）
            for i, (src_batch, tgt_batch, seg_batch, soft_tgt_batch) in enumerate(batch_loader(batch_size, src, tgt, seg, soft_tgt)):
                loss = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch)# 前向传播、反向传播、参数更新    seg_batch:torch.Size([21, 320])
                total_loss += loss.item()    # 累加损失
                lr = optimizer.param_groups[0]['lr']
                if (i + 1) % args.report_steps == 0:    # 每report_steps步打印一次训练信息
                    print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}, lr: {:.3e}".format(epoch, i + 1, total_loss / args.report_steps, lr))
                    total_loss = 0.0    # 重置损失累加器

            lr = optimizer.param_groups[0]['lr']
            result = evaluate(args, read_dataset(args, args.dev_path))  #f1
            f1 = result[0]
            
            if not os.path.exists(loss_save_path):
                os.makedirs(loss_save_path)
            with open(loss_save_path + "/loss_100.txt", "a") as f:
                f.write("epoch {:3d},valid macro f1: {:.4f}, lr: {:.4e}, batch: {:3d}\n".format(epoch, f1, lr, args.batch_size))
            if result[0] > best_result:
                best_result = result[0]
                best_result_round = epoch
                args.output_model_path = loss_save_path + "/best_model.bin"
                save_model(model, args.output_model_path)       #在验证集上找到最好的结果。
            elif epoch - best_result_round >= args.earlystop:       #连着5(args.earlystop)轮训练都没变好，就停止训练
                print("early stopping...")
                break

    # Evaluation phase.
    if args.test_path is not None:
        args.output_model_path = loss_save_path + "/best_model.bin"
        print("Test set evaluation.")
        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(torch.load(args.output_model_path))
        else:
            model.load_state_dict(torch.load(args.output_model_path))
        result = evaluate(args, read_dataset(args, args.test_path), True)        #使用测试集进行评估
        f1 = result[0]
        acc = result[1]
        pr = result[2]
        rc = result[3]
        with open(loss_save_path + "/loss_100.txt", "a") as f:
            f.write("test macro f1: {:.4f}  acc: {:.4f} pr: {:.4f} rc: {:.4f}  best_epoch:{:4d}\n".format(f1, acc, pr, rc, best_result_round))


if __name__ == "__main__":
    main()
