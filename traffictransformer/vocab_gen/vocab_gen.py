from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors, Regex
import json
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
def build_BPE(corpora_path):
    """构建BPE词汇表函数
    使用WordPiece算法构建网络流量数据的词汇表，确保词汇固定为4个字符

    参数:
        corpora_path: 语料库文件路径
    """
    num_count = 65541
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    not_change_string_count = len(special_tokens)

    source_dictionary = {}
    i = 0
    while i < num_count:
        temp_string = '{:04x}'.format(i)
        source_dictionary[temp_string] = i
        i += 1

    tokenizer = Tokenizer(models.WordPiece(
        vocab=source_dictionary,
        unk_token="[UNK]",
        max_input_chars_per_word=4
    ))

    four_char_pattern = Regex(r'.{4}')
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.BertPreTokenizer(),
        pre_tokenizers.Split(pattern=four_char_pattern, behavior="isolated")
    ])

    tokenizer.decoder = decoders.WordPiece()

    tokenizer.post_processor = processors.BertProcessing(
        sep=("[SEP]", special_tokens.index("[SEP]")),
        cls=("[CLS]", special_tokens.index("[CLS]"))
    )

    trainer = trainers.WordPieceTrainer(
        vocab_size=num_count,
        min_frequency=2,
        special_tokens=special_tokens
    )

    tokenizer.train([corpora_path, corpora_path], trainer=trainer)

    vocab = tokenizer.get_vocab()
    # filtered_vocab = {k: v for k, v in vocab.items() if len(k) == 4 or k in special_tokens}
    filtered_vocab = {k: v for k, v in vocab.items()}

    if len(filtered_vocab) < num_count:
        missing = num_count - len(filtered_vocab)
        print(f"补充 {missing} 个4字符词汇以达到目标词汇量")
        i = 0
        while len(filtered_vocab) < num_count:
            candidate = f"temp{i:03d}"
            if candidate not in filtered_vocab:
                filtered_vocab[candidate] = max(filtered_vocab.values()) + 1
            i += 1

    tokenizer.model = models.WordPiece(
        vocab=filtered_vocab,
        unk_token="[UNK]",
        max_input_chars_per_word=4
    )

    tokenizer.save("wordpiece_4chars.tokenizer.json", pretty=True)
    return 0

def build_vocab(vocab_path):
    """构建词汇表文件，只包含4个字符的词汇

    参数:
        vocab_path: 词汇表输出路径
    """
    with open("wordpiece_4chars.tokenizer.json", 'r') as f:
        vocab_json = json.load(f)

    special_tokens = ["[PAD]", "[SEP]", "[CLS]", "[UNK]", "[MASK]"]
    vocab_entries = []

    # for token in special_tokens:
    #     if token in vocab_json['model']['vocab']:
    #         vocab_entries.append(token)

    sorted_items = sorted(vocab_json['model']['vocab'].items(), key=lambda x: x[1])
    for item, _ in sorted_items:
        # if len(item) == 4 and item not in special_tokens and item not in vocab_entries:
        vocab_entries.append(item)

    with open(vocab_path, 'w') as f:
        for word in vocab_entries:
            f.write(word + "\n")

    print(f"生成的词汇表包含 {len(vocab_entries)} 个词汇")
    return 0

# build_BPE('direction_bpe_corpus_tcp_udp.txt')
build_vocab("vocab_packet_4chars.txt")
