# 处理原始数据集
HierarchicalFormer_flow/data_generation/get_finedata.py  将pcap划分为flow (已完成)

# 预训练
1. traffictransformer/vocab_gen/extractIpTcp.py 提取特征
2. traffictransformer/vocab_gen/vocab_gen_all.py 生成词表
3. traffictransformer/pre-training/preprocess.py  生成dataset
4. traffictransformer/pre-training/pretrain.py  开始预训练

# 微调
1. traffictransformer/fine-data/get_finedata.py  提取json
2. traffictransformer/fine-data/get_finetsv.py     生成tsv
3. traffictransformer/fine-tuning/run_classifier.py  得到结果