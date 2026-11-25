import csv
import random
from collections import defaultdict

class PacketSeqAugmenter:
    def __init__(self):
        self.ip_fields = [
            "timestamp",
            "direction",
            "tos",
            "len",
            "flags",
            "ttl",
            "proto",
        ]
        self.tcp_fields = [
            "dataofs",
            "flags",
            "window",
        ]
        self.udp_fields = [
            "len",
            "reserved1",
            "reserved2",
        ]
        self.ip_len = len(self.ip_fields)
        self.packet_total_fields = self.ip_len + max(len(self.tcp_fields), len(self.udp_fields))
        # self.src_port = None
        # self.dst_port = None
        # self.id_add = None
        # self.seq_low = None
        # self.ack_low = None
        self.window_add = None
        self.ip_aug_idx = [
            self.ip_fields.index(f) for f in ["timestamp", "ttl"]
        ]
        self.tcp_aug_idx = [
            self.tcp_fields.index(f) for f in ["window"]
        ]
        self.udp_aug_idx = [

        ]


    def hex2int(self, hex_str):
        return int(hex_str, 16) if hex_str not in ["[CLS]", "[SEP]"] else hex_str

    def int2hex(self, num):
        return f"{num:04x}"

    def parse_seq_format(self, seq_str):
        tokens = seq_str.strip().split()
        if tokens[0] != "[CLS]":
            raise ValueError(f"序列必须以[CLS]开头，当前开头：{tokens[0]}")
        total_tokens = len(tokens)
        packet_sep_total = self.packet_total_fields + 1
        packet_num = total_tokens // packet_sep_total
        packets = []
        seps = []
        ptr = 1
        for _ in range(packet_num):
            packet_fields = tokens[ptr:ptr + self.packet_total_fields]
            if len(packet_fields) != self.packet_total_fields:
                raise ValueError(f"包头字段数异常：应为{self.packet_total_fields}个，实际{len(packet_fields)}个")
            packets.append(packet_fields)

            sep = tokens[ptr + self.packet_total_fields]
            if sep != "[SEP]":
                raise ValueError(f"包头后必须是[SEP]，当前是：{sep}")
            seps.append(sep)

            ptr += self.packet_total_fields + 1

        if ptr != len(tokens):
            raise ValueError(f"序列格式异常：解析后剩余{len(tokens)-ptr}个多余token")

        return {
            "cls": "[CLS]",
            "packets": packets,
            "seps": seps,
            "raw_tokens": tokens
        }

    def augment_single_packet(self, packet_fields, packet_idx):
        random.seed(7)
        augmented = packet_fields.copy()
        ip_fields = augmented[:self.ip_len]
        trans_fields = augmented[self.ip_len:]

        for ip_idx in self.ip_aug_idx:
            hex_val = ip_fields[ip_idx]
            int_val = self.hex2int(hex_val)
            if self.ip_fields[ip_idx] == "timestamp":
                if packet_idx == 0:
                    continue
                int_val = max(0, int_val + random.randint(-1*max(30, int_val), 30))
                int_val = min(2**16-1, int_val)
            elif self.ip_fields[ip_idx] == "ttl":
                if random.random() < 0.9:
                    int_val = int_val
                else:
                    int_val = random.choice([45, 64, 115, 50, 21])
            ip_fields[ip_idx] = self.int2hex(int_val)
        augmented[:self.ip_len] = ip_fields

        proto_hex = ip_fields[self.ip_fields.index("proto")]
        proto_val = self.hex2int(proto_hex)
        if proto_val == 6:
            for tcp_idx in self.tcp_aug_idx:
                if tcp_idx >= len(trans_fields):
                    continue
                hex_val = trans_fields[tcp_idx]
                int_val = self.hex2int(hex_val)
                if self.tcp_fields[tcp_idx] == "window":
                    if not self.window_add:
                        self.window_add = random.randint(-1*min(100,int_val), 100)
                    int_val = int_val+self.window_add
                    int_val = random.randint(0, 100) if int_val < 0 else int_val
                    int_val = int_val%(2**16)
                    # int_val = random.randint(0, 65535)
                trans_fields[tcp_idx] = self.int2hex(int_val)
        elif proto_val == 17:
            for udp_idx in self.udp_aug_idx:
                if udp_idx >= len(trans_fields):
                    continue
                hex_val = trans_fields[udp_idx]
                int_val = self.hex2int(hex_val)
                trans_fields[udp_idx] = self.int2hex(int_val)
        augmented[self.ip_len:] = trans_fields

        return augmented

    def augment_seq_sample(self, seq_str):
        parsed = self.parse_seq_format(seq_str)
        augmented_packets = []
        for idx in range(len(parsed["packets"])):
            pkt = parsed["packets"][idx]
            augmented_packets.append(self.augment_single_packet(pkt, idx))
        # augmented_packets = [self.augment_single_packet(pkt) for pkt in parsed["packets"]]
        # self.src_port = None
        # self.dst_port = None
        # self.id_add = None
        # self.seq_low = None
        # self.ack_low = None
        self.window_add = None
        cz_tokens = [parsed["cls"]]
        for pkt, sep in zip(augmented_packets, parsed["seps"]):
            cz_tokens.extend(pkt)
            cz_tokens.append(sep)
        if len(cz_tokens) != len(parsed["raw_tokens"]):
            raise ValueError(f"重组后格式异常：原始{len(parsed['raw_tokens'])}个token，重组后{len(cz_tokens)}个")
        return " ".join(cz_tokens)

    def balance_seq_dataset(self, data, labels):
        label2seqs = defaultdict(list)
        for seq, lbl in zip(data, labels):
            label2seqs[lbl].append(seq)
        max_count = max(len(seqs) for seqs in label2seqs.values())
        max_count = max(500, max_count)
        print(f"类别平衡目标：每个类别{max_count}个样本（当前最多类样本数）")

        balanced_data = []
        balanced_labels = []
        for lbl, seqs in label2seqs.items():
            current_count = len(seqs)
            if current_count >= max_count:
                selected = random.sample(seqs, max_count)
            else:
                selected = seqs.copy()
                need = max_count - current_count
                for i in range(need):
                    base_seq = seqs[i % current_count]
                    augmented_seq = self.augment_seq_sample(base_seq)
                    selected.append(augmented_seq)
            balanced_data.extend(selected)
            balanced_labels.extend([lbl] * len(selected))

        print(f"平衡后总样本数：{len(balanced_data)}（类别数：{len(label2seqs)}）")
        return balanced_data, balanced_labels


def write_balanced_seq_tsv(data, labels, file_dir, type, ip_fields, tcp_fields, udp_fields):
    augmenter = PacketSeqAugmenter(ip_fields, tcp_fields, udp_fields)
    try:
        _ = augmenter.parse_seq_format(data[0])
        print("✅ 第一个样本格式验证通过")
    except Exception as e:
        raise ValueError(f"❌ 样本格式错误：{str(e)}") from e
    balanced_data, balanced_labels = augmenter.balance_seq_dataset(data, labels)

    tsv_data = [["label", "text_a"]]
    for lbl, seq in zip(balanced_labels, balanced_data):
        tsv_data.append([str(lbl), seq])

    output_path = f"{file_dir.rstrip('/')}/{type}_balanced_dataset.tsv"
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(tsv_data)

    print(f"✅ 已写入平衡数据集：{output_path}")
    print(f"📊 平衡后统计：{len(balanced_labels)}个样本，{len(set(balanced_labels))}个类别")
    return output_path