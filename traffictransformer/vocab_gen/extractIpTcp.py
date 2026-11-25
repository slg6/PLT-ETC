import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys
from scapy.all import rdpcap, IP, TCP, UDP, PacketList
# import json
import tqdm
import multiprocessing as mp


class DirectionBasedHeaderProcessor:
    def __init__(self):
        self.standard_length = 4
        self.timestamp_standard_length = 4
        self.timestamp_max = 0xFFFF
        # --------------------------
        # --------------------------
        self.ip_fields = [
            "timestamp",

            "direction",




            "tos",

            "len",


            "flags",


            "ttl",

            "proto",
        ]

        # --------------------------
        # --------------------------
        self.tcp_fields = [



            "dataofs",


            "flags",

            "window",



        ]

        # --------------------------
        # --------------------------
        self.udp_fields = [


            "len",
            "reserved1",
            "reserved2",
        ]

        self.extracted_headers = []
        self.prev_timestamp = None

        self.base_src_ip = None
        self.base_dst_ip = None
        self.base_sport = None
        self.base_dport = None
        self.base_proto = None

        self.dir_hex_map = {
            "OUT": "0001",
            "IN": "0002"
        }

        self.proto_map = {
            6: "TCP",
            17: "UDP"
        }
    # --------------------------
    # --------------------------
    def _standardize_timestamp(self, timestamp_diff_ms):
        if timestamp_diff_ms < 0:
            timestamp_diff_ms = 0
        elif timestamp_diff_ms > self.timestamp_max:
            timestamp_diff_ms = self.timestamp_max

        hex_str = f"{int(timestamp_diff_ms):x}"
        if len(hex_str) < self.timestamp_standard_length:
            return hex_str.zfill(self.timestamp_standard_length)
        return hex_str[-self.timestamp_standard_length:]
    # --------------------------
    # --------------------------
    def _calc_timestamp_diff(self, current_packet_time):
        current_ts_ms = int(round(current_packet_time * 1000))
        if self.prev_timestamp is None:
            self.prev_timestamp = current_ts_ms
            return 0

        diff_ms = current_ts_ms - self.prev_timestamp
        self.prev_timestamp = current_ts_ms
        return diff_ms
    # --------------------------
    # --------------------------
    def _determine_direction(self, ip_layer, transport_layer, proto):
        """
        以当前流的第一个包为基准，判断后续包的方向（增加协议校验）
        proto: 传输层协议（"TCP"或"UDP"）
        """
        if self.base_src_ip is None:
            self.base_src_ip = ip_layer.src
            self.base_dst_ip = ip_layer.dst
            self.base_sport = transport_layer.sport
            self.base_dport = transport_layer.dport
            self.base_proto = proto
            return "OUT"

        if proto != self.base_proto:
            return "UNKNOWN"

        if (ip_layer.src == self.base_src_ip and
            ip_layer.dst == self.base_dst_ip and
            transport_layer.sport == self.base_sport and
            transport_layer.dport == self.base_dport):
            return "OUT"
        elif (ip_layer.src == self.base_dst_ip and
              ip_layer.dst == self.base_src_ip and
              transport_layer.sport == self.base_dport and
              transport_layer.dport == self.base_sport):
            return "IN"
        else:
            return "UNKNOWN"


    # --------------------------
    # --------------------------
    def _standardize_field(self, value, is_numeric=True):
        if is_numeric:
            hex_str = f"{value:x}"
        else:
            hex_str = self.dir_hex_map.get(value, "0000") if value != "UNKNOWN" else "0000"

        if len(hex_str) > self.standard_length:
            return hex_str[-self.standard_length:]
        return hex_str.zfill(self.standard_length)


    # --------------------------
    # --------------------------
    def _process_ip_header(self, ip_layer, transport_layer, proto, timestamp_diff_ms):
        ip_data = {}
        # ip_data["version"] = self._standardize_field(ip_layer.version)
        # ip_data["headlen"] = self._standardize_field(ip_layer.ihl)
        ip_data["tos"] = self._standardize_field(ip_layer.tos)
        ip_data["len"] = self._standardize_field(ip_layer.len)
        # ip_data["id"] = self._standardize_field(ip_layer.id)
        ip_data["flags"] = self._standardize_field(ip_layer.flags.value)
        # ip_data["frag"] = self._standardize_field(ip_layer.frag)
        ip_data["ttl"] = self._standardize_field(ip_layer.ttl)
        ip_data["proto"] = self._standardize_field(ip_layer.proto)
        # ip_data["chksum"] = self._standardize_field(ip_layer.chksum)
        direction = self._determine_direction(ip_layer, transport_layer, proto)
        ip_data["direction"] = self._standardize_field(direction, is_numeric=False)
        ip_data["timestamp"] = self._standardize_timestamp(timestamp_diff_ms)
        return ip_data


    # --------------------------
    # --------------------------
    def _process_tcp_header(self, tcp_layer):
        tcp_data = {}
        # tcp_data["sport"] = self._standardize_field(tcp_layer.sport)
        # tcp_data["dport"] = self._standardize_field(tcp_layer.dport)
        # tcp_data["seq_low"] = self._standardize_field(tcp_layer.seq & 0xFFFF)
        # tcp_data["ack_low"] = self._standardize_field(tcp_layer.ack & 0xFFFF)
        tcp_data["dataofs"] = self._standardize_field(tcp_layer.dataofs)
        # tcp_data["res"] = self._standardize_field(tcp_layer.reserved)
        tcp_data["flags"] = self._standardize_field(tcp_layer.flags.value)
        tcp_data["window"] = self._standardize_field(tcp_layer.window)
        # tcp_data["chksum"] = self._standardize_field(tcp_layer.chksum)
        # tcp_data["urgptr"] = self._standardize_field(tcp_layer.urgptr)
        return tcp_data


    # --------------------------
    # --------------------------
    def _process_udp_header(self, udp_layer):
        udp_data = {}
        # udp_data["sport"] = self._standardize_field(udp_layer.sport)
        # udp_data["dport"] = self._standardize_field(udp_layer.dport)
        udp_data["len"] = self._standardize_field(udp_layer.len)
        udp_data["reserved1"] = self._standardize_field(0)
        udp_data["reserved2"] = self._standardize_field(0)
        # udp_data["reserved3"] = self._standardize_field(0)
        # udp_data["reserved4"] = self._standardize_field(0)
        # udp_data["reserved5"] = self._standardize_field(0)
        # udp_data["reserved6"] = self._standardize_field(0)
        return udp_data


    # --------------------------
    # --------------------------
    def _generate_grouped_string(self, ip_data, transport_data, proto):
        """
        生成符合格式的字符串（支持TCP/UDP）
        proto: 传输层协议（"TCP"或"UDP"）
        """
        full_hex = ""
        for field in self.ip_fields:
            full_hex += ip_data[field]
        if proto == "TCP":
            for field in self.tcp_fields:
                full_hex += transport_data[field]
        elif proto == "UDP":
            for field in self.udp_fields:
                full_hex += transport_data[field]

        hex_groups = [full_hex[i:i+4] for i in range(0, len(full_hex), 4) if full_hex[i:i+4]]
        return " ".join(hex_groups)


    # --------------------------
    # --------------------------
    def _read_pcap(self, pcap_input):
        if isinstance(pcap_input, str):
            return rdpcap(pcap_input)
        elif isinstance(pcap_input, list) or isinstance(pcap_input, PacketList):
            return pcap_input
        else:
            return list([pcap_input])

    def extract_headers_from_pcap(self, pcap_file, max_packets=None):
        try:
            packets = self._read_pcap(pcap_file)
            count = 0
            total_packets = 0
            filtered_packets = 0
            self.__reset_stream()

            for packet in packets:
                if max_packets and count >= max_packets:
                    break
                total_packets += 1
                try:
                    if packet.haslayer(IP):
                        ip_layer = packet.getlayer(IP)
                        if ip_layer.version != 4:
                            filtered_packets += 1
                            continue
                        proto = ip_layer.proto

                        if proto == 6 and packet.haslayer(TCP):
                            transport_layer = packet.getlayer(TCP)
                            transport_proto = "TCP"
                        elif proto == 17 and packet.haslayer(UDP):
                            transport_layer = packet.getlayer(UDP)
                            transport_proto = "UDP"
                        else:
                            filtered_packets += 1
                            continue
                    else:
                        filtered_packets += 1
                        continue

                    timestamp_diff_ms = self._calc_timestamp_diff(packet.time)
                    ip_data = self._process_ip_header(ip_layer, transport_layer, transport_proto, timestamp_diff_ms)
                    if transport_proto == "TCP":
                        transport_data = self._process_tcp_header(transport_layer)
                    else:
                        transport_data = self._process_udp_header(transport_layer)

                    if ip_data["direction"] == "0000" and count > 0:
                        print(f"警告：{pcap_file}的第{count+1}个{transport_proto}包方向未知，已跳过")
                        continue

                    grouped_str = self._generate_grouped_string(ip_data, transport_data, transport_proto)
                    self.extracted_headers.append(grouped_str)
                    count += 1

                except Exception as e:
                    print(f"解析{pcap_file}的第{total_packets}个包出错: {str(e)}，已跳过")
                    filtered_packets += 1

            if isinstance(pcap_file, str):
                print(f"从{pcap_file}（1个流）中提取了{count}个有效IP+TCP/UDP包头")
                print(f"  总包数: {total_packets}, 过滤/错误包数: {filtered_packets}")
            # else:
            return count

        except Exception as e:
            print(f"处理{pcap_file}出错: {str(e)}")
            return 0


    # --------------------------
    # --------------------------
    def __reset_stream(self):
        self.extracted_headers = []
        self.base_src_ip = None
        self.base_dst_ip = None
        self.base_sport = None
        self.base_dport = None
        self.base_proto = None
        self.prev_timestamp = None


    # --------------------------
    # --------------------------
    def generate_bpe_corpus(self, output_file, is_multi=False):
        if is_multi:
            pid = os.getpid()
            file_path = f"{output_file}_{pid}.txt"
        else:
            file_path = output_file

        with open(file_path, "a", encoding="utf-8") as f:
            is_first_packet = True
            for packet_str in self.extracted_headers:
                if is_first_packet:
                    f.write(f"\n||{packet_str}\n")
                    is_first_packet = False
                else:
                    f.write(f"{packet_str}\n")


# --------------------------
# --------------------------
def extractPcap(pcap_file, corpus_prefix="direction_bpe_corpus", is_multi=False):
    processor = DirectionBasedHeaderProcessor()
    processor.extract_headers_from_pcap(pcap_file, max_packets=30)
    processor.generate_bpe_corpus(corpus_prefix, is_multi=is_multi)


def merge_temp_files(corpus_prefix, temp_dir):
    final_corpus_path = f"{corpus_prefix}.txt"
    temp_files = [f for f in os.listdir(temp_dir) if f.startswith(f"{corpus_prefix}_") and f.endswith(".txt")]

    with open(final_corpus_path, "w", encoding="utf-8") as final_f:
        for temp_file in temp_files:
            temp_path = os.path.join(temp_dir, temp_file)
            with open(temp_path, "r", encoding="utf-8") as temp_f:
                final_f.write(temp_f.read())
            os.remove(temp_path)

    print(f"合并完成！最终语料库路径：{final_corpus_path}")
    print(f"合并的临时文件数量：{len(temp_files)}")


def pretrain_dataset_generation(output_split_path, corpus_prefix="direction_bpe_corpus", is_multi=True):
    # pcap_dir = os.path.join(output_split_path, "splitcap")
    pcap_dir = output_split_path
    if not os.path.exists(pcap_dir):
        raise FileNotFoundError(f"PCAP目录不存在：{pcap_dir}")

    all_pcap_files = []
    for root, _, files in os.walk(pcap_dir):
        for file in files:
            if file.lower().endswith((".pcap", ".pcapng")):
                all_pcap_files.append(os.path.join(root, file))

    if not all_pcap_files:
        print("未找到任何PCAP文件！")
        return

    print(f"共发现{len(all_pcap_files)}个PCAP文件（每个对应1个流）")

    if is_multi:
        temp_dir = f"{corpus_prefix}_temp"
        os.makedirs(temp_dir, exist_ok=True)
        corpus_temp_prefix = os.path.join(temp_dir, corpus_prefix)

        pbar = tqdm.tqdm(total=len(all_pcap_files), desc="多进程提取包头（TCP+UDP）")
        update_pbar = lambda *args: pbar.update()

        pool = mp.Pool(processes=min(mp.cpu_count(), 40))
        for pcap_file in all_pcap_files:
            pool.apply_async(extractPcap, args=(pcap_file, corpus_temp_prefix, True), callback=update_pbar)

        pool.close()
        pool.join()
        pbar.close()

        print("\n开始合并多进程临时文件...")
        merge_temp_files(corpus_prefix, temp_dir)
        os.rmdir(temp_dir)

    else:
        final_corpus_path = f"{corpus_prefix}.txt"
        if os.path.exists(final_corpus_path):
            os.remove(final_corpus_path)

        for pcap_file in tqdm.tqdm(all_pcap_files, desc="单进程提取包头（TCP+UDP）"):
            extractPcap(pcap_file, final_corpus_path, is_multi=False)

        print(f"单进程处理完成！最终语料库路径：{final_corpus_path}")


if __name__ == "__main__":
    pretrain_dataset_generation(
        output_split_path=os.path.expanduser("~/llmfortraffic/dataPcap_feature/cic"),
        corpus_prefix="direction_bpe_corpus_tcp_udp",
        is_multi=True
    )