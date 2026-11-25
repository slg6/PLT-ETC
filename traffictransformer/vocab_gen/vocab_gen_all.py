import itertools
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
def generate_hex_4char_vocab(vocab_save_path="vocab_packet_4chars_all.txt"):
    hex_chars = "0123456789abcdef"
    vocab = ["[PAD]", "[SEP]", "[CLS]", "[UNK]", "[MASK]"]
    for chars in itertools.product(hex_chars, repeat=4):
        token = "".join(chars)
        vocab.append(token)

    assert len(vocab) == 65541, f"词表大小错误：实际{len(vocab)}个，预期65536个"

    with open(vocab_save_path, "w", encoding="utf-8") as f:
        for idx, token in enumerate(vocab):
            f.write(f"{token}\n")

    print(f"词表生成完成！共{len(vocab)}个token，已保存至{vocab_save_path}")
    return vocab

if __name__ == "__main__":
    generate_hex_4char_vocab()
