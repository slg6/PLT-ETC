import os
import random
import shutil
import binascii
import scapy.all as scapy
from functools import reduce
from flowcontainer.extractor import extract
from utils import *
import json,operator
from tqdm import tqdm
import pickle
import multiprocessing as mp
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
from sklearn.model_selection import train_test_split
import string
#scapy.load_layer("tls")
from scapy.layers.tls.handshake import TLSClientHello,TLSServerHello
#from scapy.layers.tls.extensions import TLS_Ext_ServerName
import sys

# sys.path.append(os.path.expanduser("~/llmfortraffic/looptransformer/traffictransformer"))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vocab_gen.extractIpTcp import DirectionBasedHeaderProcessor as DBHProcessor

def random_tcp_ts_option(packets):
    src_ts = None
    dst_ts = None
    random_src_ts = random_field(32)
    random_dst_ts = random_field(32)
    src_port = None
    for packet in packets:
        if packet.haslayer(scapy.TCP):
            tcp_options = [list(option) for option in packet['TCP'].options]
            for option in tcp_options:
                if option[0] == 'Timestamp':
                    if src_port==None:
                        src_port = packet['TCP'].sport
                        src_ts = option[1][0]
                        if option[1][1]!=0:
                            dst_ts = option[1][1]
                    if packet['TCP'].sport == src_port:
                        if option[1][1]!=0:
                            option[1] = (random_src_ts + option[1][0]-src_ts, random_dst_ts + option[1][1]-dst_ts)
                        else:
                            option[1] = (random_src_ts + option[1][0]-src_ts, 0)
                    else:
                        if dst_ts==None:
                            dst_ts = option[1][0]
                        if option[1][1]!=0:
                            option[1] = (random_dst_ts + option[1][0]-dst_ts, random_src_ts + option[1][1]-src_ts)
                        else:
                            option[1] = (random_dst_ts + option[1][0]-dst_ts, 0)

            packet['TCP'].options = [tuple(option) for option in tcp_options]
    return packets

def random_tls_sni():
    # We're using few bytes and won't get to the sni field
    # server name(sni) is a variable-length field, modifying it triggers a series of length modifications, ranging from tls extension length, tls length, IP length, and even packet ack.
    scapy.load_layer("tls")
    from scapy.layers.tls.handshake import TLSClientHello,TLSServerHello
    from scapy.layers.tls.extensions import TLS_Ext_ServerName
    packets = scapy.rdpcap("1.pcap")
    count = 0
    for packet in packets:
        # change server name
        print(count)
        count+=1
        if packet.haslayer(TLSClientHello):
            tls_client_hello = packet[TLSClientHello]
            tls_client_hello.show()
            print(packet['IP'].len,packet[TLS].len,tls_client_hello.msglen,tls_client_hello.extlen)
            add = 0
            if tls_client_hello.ext==None:
                continue
            for ext in tls_client_hello.ext:
                # Check for Server Name extension
                if isinstance(ext, TLS_Ext_ServerName):
                    for server_name in ext.servernames:
                        random_length = random.randint(10, 25) #random length
                        #random_length = server_name.namelen #keep length
                        print(random_length)
                        # ramdom server name
                        random_server_name = ''.join(random.choices(string.ascii_lowercase + string.digits, k=random_length))
                        print(random_server_name)
                        server_name.servername = random_server_name.encode("UTF-8")
                        add += random_length - server_name.namelen
                        server_name.namelen = random_length
                        server_name.nametype = 0

                    print("add: ", add)
                    ext.len += add
                    ext.servernameslen += add

            tls_client_hello.extlen +=add
            tls_client_hello.msglen +=add
            packet[TLS].len += add
            packet['IP'].len += add

        #change tls1.2 timestamps
        if packet.haslayer(TLSClientHello):
            packet[TLSClientHello].gmt_unix_time = random_field(32)
        if packet.haslayer(TLSServerHello):
            packet[TLSServerHello].gmt_unix_time = random_field(32)

    scapy.wrpcap("random_sni.pcap", packets)

def random_ip_port(packets:scapy.PacketList):
    if len(packets) > 0:
        if scapy.IP in packets[0]:
            client_ip = packets[0][scapy.IP].src
            server_ip = packets[0][scapy.IP].dst
            r_client_ip = random_ipv4()
            r_server_ip = random_ipv4()
        elif scapy.IPv6 in packets[0]:
            client_ip = packets[0][scapy.IPv6].src
            server_ip = packets[0][scapy.IPv6].dst
            r_client_ip = random_ipv6()
            r_server_ip = random_ipv6()

        else:
            print("other L3 protocol")
        r_client_port = random_field(16)
        r_server_port = random_field(16)
    else:
        print("No packet in the pcap file")
        return -1

    for packet in packets:
        if scapy.IP in packet:
            is_client = packet[scapy.IP].src==client_ip
            packet[scapy.IP].src = r_client_ip if is_client else r_server_ip
            packet[scapy.IP].dst = r_server_ip if is_client else r_client_ip
            if scapy.TCP in packet:
                packet[scapy.TCP].sport = r_client_port if is_client else r_server_port
                packet[scapy.TCP].dport = r_server_port if is_client else r_client_port
            if scapy.UDP in packet:
                packet[scapy.UDP].sport = r_client_port if is_client else r_server_port
                packet[scapy.UDP].dport = r_server_port if is_client else r_client_port
        if scapy.IPv6 in packet:
            is_client = packet[scapy.IPv6].src==client_ip
            packet[scapy.IPv6].src = r_client_ip if is_client else r_server_ip
            packet[scapy.IPv6].dst = r_server_ip if is_client else r_client_ip
            if scapy.TCP in packet:
                packet[scapy.TCP].sport = r_client_port if is_client else r_server_port
                packet[scapy.TCP].dport = r_server_port if is_client else r_client_port
            if scapy.UDP in packet:
                packet[scapy.UDP].sport = r_client_port if is_client else r_server_port
                packet[scapy.UDP].dport = r_server_port if is_client else r_client_port

    return packets

def random_tls_randomtime(packets:scapy.PacketList):
    for packet in packets:
        if packet.haslayer(TLSClientHello):
            packet[TLSClientHello].gmt_unix_time = random_field(32)
        if packet.haslayer(TLSServerHello):
            packet[TLSServerHello].gmt_unix_time = random_field(32)
    return packets

def extract_flow_features(flow_path, enhance_factor=1, max_packets=20, min_flow=5):
    try:
        # original_packets = scapy.rdpcap(flow_path)
        original_packets = []
        n = 30
        with scapy.PcapReader(flow_path) as reader:
            for i, pkt in enumerate(reader):
                if i >= n:
                    break
                original_packets.append(pkt)
        if len(original_packets) < min_flow:
            return None
        original_packets = original_packets[:max_packets]
        enhanced_features = {}
        # for enhance_id in range(enhance_factor):
        for enhance_id in range(enhance_factor):
            packets = original_packets.copy()
            flow_features = {}
            packet_index = 1
            # if enhance_id > 0:
            try:
                packets = random_ip_port(packets)
            except:
                print(f"Warning: Error in random_ip_port for {flow_path}: {e}")
            try:
                packets = random_tcp_ts_option(packets)
            except:
                print(f"Warning: Error in random_tcp_ts_option for {flow_path}: {e}")
            try:
                packets = random_tls_randomtime(packets)
            except:
                print(f"Warning: Error in random_tls_randomtime for {flow_path}: {e}")

            dbh = DBHProcessor()
            dbh.extract_headers_from_pcap(packets)
            for i, packet in enumerate(tqdm(dbh.extracted_headers, desc="提取特征", unit="包", leave=False)):
                flow_features[str(packet_index)] = [packet]
                packet_index += 1
            if len(flow_features) == 0:
                continue
            enhanced_features[f"enhance_{enhance_id}"] = flow_features
        return enhanced_features

    except Exception as e:
        print(f"Error processing {flow_path}: {e}")
        return None

def generation_multiP(dataset_path, samples, enhance_factor=1, max_packets=20, min_flow=5, traceMaxFlow=1000):
    output_file = os.path.join(dataset_path, f"{os.path.basename(dataset_path)}_codedata_enhance{enhance_factor}.json") #'../../dataPcap/idi/idi_codedata.json'
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            dataset = json.load(f)
    else:
        dataset = {}

    categories = [cat for cat in os.listdir(dataset_path)
                 if os.path.isdir(os.path.join(dataset_path, cat))]
    for category in tqdm(categories, desc="处理类别", unit="类"):
    # for category in os.listdir(dataset_path):   #category:'WeMoSwitch2'
        category_path = os.path.join(dataset_path, category)    #'../../dataPcap/idi/WeMoSwitch2'
        if not os.path.isdir(category_path):
            continue

        # print(f"Processing category: {category}")
        if category not in dataset:
            dataset[category] = {}

        traces = [trace for trace in os.listdir(category_path) if trace.startswith('flow_')]
        # for trace in os.listdir(category_path): #trace:'flow_Setup-C-2-STA'
        #     if not trace.startswith('flow_'):
        #         continue
        echo = 0
        batch = 10000
        processed_flows = 0
        for trace in tqdm(traces, desc=f"处理{category}的踪迹", unit="踪迹", leave=False):
            flag = False
            if trace in dataset[category]:
                print(f"跳过已处理的踪迹: {category}/{trace}")
                processed_flows += 1
                continue
            trace_path = os.path.join(category_path, trace) #'../../dataPcap/idi/WeMoSwitch2/flow_Setup-C-2-STA'

            if processed_flows >= samples:
                break

            flows = [f for f in os.listdir(trace_path) if f.endswith('.pcap')]
            if len(flows) == 0:
                continue
            # for flow in os.listdir(trace_path): #'Setup-C-2-STA.pcap.TCP_10-22-22-2_38785_54-243-187-195_443.pcap'
            #     if not flow.endswith('.pcap'):
            #         continue
            dataset[category][trace] = {}
            trace_flow = 0
            for flow in tqdm(flows, desc=f"处理{trace}的流文件", unit="流", leave=False):
                if trace_flow >= traceMaxFlow:
                    break
                flow_path = os.path.join(trace_path, flow)  #'../../dataPcap/idi/WeMoSwitch2/flow_Setup-C-2-STA/Setup-C-2-STA.pcap.TCP_10-22-22-2_38785_54-243-187-195_443.pcap'
                features = extract_flow_features(
                    flow_path,
                    enhance_factor,
                    max_packets,
                    min_flow
                )
                if features:
                    flag = True
                    trace_flow += 1
                    for enhance_id, enhanced_feature in features.items():
                        flow_name = f"{flow}_{enhance_id}"
                        dataset[category][trace][flow_name] = enhanced_feature
            if flag:
                processed_flows += 1
            echo += 1
            if echo > batch:
                print(f"\n保存处理进度: {category}/{trace}")
                with open(output_file, 'w') as f:
                    json.dump(dataset, f, indent=4)
                echo = 0

    # output_file = os.path.join(dataset_path, f"{os.path.basename(dataset_path)}_data.json") #'../../dataPcap/idi/idi_data.json'
    # with open(output_file, 'w') as f:
    #     json.dump(dataset, f, indent=4)
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=4)
    print(f"Dataset saved to: {output_file}")

    return dataset


def convert_splitcap(pcapng_path, is_pcap_label=False):
    # pcapng_path: the path of pcapng files (if the traffic is the pacp type, pcapng_path = pcap_path)
    # pcap_path: the path of pcap files
    # pcap_split_path: the path of splited pcap files
    # pcapng to pcap
    for parent, dirs, files in os.walk(pcapng_path):
        for file in files:
            if file.endswith('.pcapng'):
                input_file = os.path.join(parent, file)
                output_file = os.path.join(parent, file.replace('.pcapng', '.pcap'))
                cmd = f"editcap -F pcap {input_file} {output_file}"
                print(f"执行转换: {cmd}")
                os.system(cmd)
                os.remove(input_file)
            else:
                continue
                # input_file = os.path.join(parent, file)
                # output_file = os.path.join(parent, file)
                # print(input_file, "    ", output_file)
    label_dirs = []
    for parent, dirs, files in os.walk(pcapng_path):
        label_dirs.extend(dirs)     #['WeMoSwitch2', 'D-LinkHomeHub', 'HomeMaticPlug', 'WeMoLink', 'HueBridge', 'WeMoInsightSwitch2', 'D-LinkSensor', 'MAXGateway', 'Withings', 'D-LinkDoorSensor', 'EdnetGateway', 'Aria', 'D-LinkCam', 'EdimaxPlug1101W', 'SmarterCoffee', 'EdnetCam1', 'D-LinkSwitch', 'D-LinkDayCam', 'EdimaxCam2', 'D-LinkWaterSensor', 'WeMoSwitch', 'D-LinkSiren', 'TP-LinkPlugHS110', 'TP-LinkPlugHS100', 'HueSwitch', 'EdnetCam2', 'WeMoInsightSwitch', 'Lightify', 'iKettle2', 'EdimaxPlug2101W', 'EdimaxCam1']
        break
    print(f"发现类别数: {len(label_dirs)}") #31
    # split pcap
    for label_dir in label_dirs:
        label_path = os.path.join(pcapng_path, label_dir)   #'../../dataPcap/idi/WeMoSwitch2'
        for parent, dirs, files in os.walk(label_path): #parent:'../../dataPcap/idi/WeMoSwitch2' dirs:[] files:['Setup-C-5-STA.pcap', 'Setup-C-4-STA.pcap', 'Setup-C-1-STA.pcap', 'Setup-C-2-STA.pcap', '_iotdevice-mac.txt', 'Setup-C-3-STA.pcap']
            for file in files:
                if file.endswith('.pcap'):
                    flow_dir = os.path.join(label_path, f"flow_{file[:-5]}")    #'../../dataPcap/idi/WeMoSwitch2/flow_Setup-C-5-STA'
                    input_pcap = os.path.join(parent, file)#'../../dataPcap/idi/WeMoSwitch2/Setup-C-5-STA.pcap'
                    if os.path.exists(flow_dir) and os.listdir(flow_dir):
                        print(f"目标文件夹{flow_dir}不为空，跳过分割")
                        os.remove(input_pcap)
                        print(f"删除原始pcap文件: {input_pcap}")
                        continue

                    os.makedirs(flow_dir, exist_ok=True)

                    split_cap(input_pcap, output_dir=flow_dir)

                    for flow_file in os.listdir(flow_dir):
                        flow_path = os.path.join(flow_dir, flow_file)#'../../dataPcap/idi/WeMoSwitch2/flow_Setup-C-4-STA/Setup-C-4-STA.pcap.TCP_10-22-22-2_42527_173-194-72-188_443.pcap'
                        file_size = float(size_format(os.path.getsize(flow_path)))

                        # if file_size < 2:
                        #     os.remove(flow_path)
                        #     continue

                        if file_size > 10240:
                            print(f"分割大文件: {flow_path}")
                            cmd = f"editcap -i 300 {flow_path} {flow_path}"
                            os.system(cmd)
                    os.remove(input_pcap)
                    print(f"删除原始pcap文件: {input_pcap}")
            break
    # remove class that has less flow
    all_flows = []
    removed_dirs = []

    for label_dir in label_dirs:
        label_path = os.path.join(pcapng_path, label_dir)#'../../dataPcap/idi/WeMoSwitch2'
        flow_dirs = [d for d in os.listdir(label_path) if d.startswith('flow_')]#['flow_Setup-C-2-STA', 'flow_Setup-C-1-STA', 'flow_Setup-C-3-STA', 'flow_Setup-C-4-STA', 'flow_Setup-C-5-STA']
        total_flows = 0

        for flow_dir in flow_dirs:
            flow_path = os.path.join(label_path, flow_dir)#'../../dataPcap/idi/WeMoSwitch2/flow_Setup-C-2-STA'
            flow_count = len([f for f in os.listdir(flow_path) if f.endswith('.pcap')])#25
            total_flows += flow_count

        print(f"{label_dir}: {total_flows} 个样本")

        if total_flows < 10:
            print(f"移除类别 {label_dir}: 样本数量({total_flows})不足10")
            shutil.rmtree(label_path)
            removed_dirs.append(label_dir)
        else:
            all_flows.append(total_flows)   #[129, 885, 20, 1035, 3677, 76, 688, 97, 112, 123, 179, 100, 377, 261, 42, 33, 611, 135, 45, 678, 318, 675, 154, 160, 3383, 28, 582, 120, 40, 235, 67]

    for dir_name in removed_dirs:
        label_dirs.remove(dir_name)

    print(f"处理完成! 总样本数: {sum(all_flows)}, 剩余类别数: {len(label_dirs)}")

def dataset_extract(dataset_save_path, features, train_rate=0.8, samples=1000, et=False):
    print("read dataset from json file.")
    # with open(dataset_save_path + "/dataset.json","r") as f:

    with open(dataset_save_path + "/data.json","r") as f:
        dataset = json.load(f)

    for app_label in dataset.keys():
        if samples == 0:
            dataset[app_label]["datagram"] = dict(list(dataset[app_label]["datagram"].items()))
            dataset[app_label]["samples"] = dataset[app_label]["samples"]
        else:
            dataset[app_label]["datagram"] = dict(list(dataset[app_label]["datagram"].items())[:samples])
            dataset[app_label]["samples"] = min(samples, dataset[app_label]["samples"])
    #slg
    dataset_statistic = [0] * len(dataset.keys())
    #slg
    data_all = []
    for app_label in dataset.keys():
        # for index_sample in range(len(dataset[app_label]["length"])):
        for index_sample in range(dataset[app_label]["samples"]):
            x = []
            flag = True
            for feature in features:
                if len(dataset[app_label]['datagram'][str(index_sample+1)]) == 0:
                    flag = False
                    continue
                x.append(dataset[app_label][feature][str(index_sample+1)])
            if flag:
                x.append(int(app_label))
                dataset_statistic[int(app_label)]+=1
                data_all.append(x)
    data = pd.DataFrame(data_all,columns=features+['label'])


    print("category flow")
    for index in range(len(dataset_statistic)):
        print("%s\t%d" % (index, dataset_statistic[index]))
    print("all\t%d" % (sum(dataset_statistic)))

    data_train, data_test = train_test_split(data, test_size=1-train_rate, random_state=41,stratify=data["label"])
    # split validate set and test set
    data_val, data_test = train_test_split(data_test, test_size=0.5, random_state=42,stratify=data_test["label"])

    print("label number of train: {}, val: {}, test: {}.".format(len(data_train['label'].value_counts()), len(data_val['label'].value_counts()),len(data_test['label'].value_counts()) ))
    data_train = data_train.reset_index(drop=True)
    data_val = data_val.reset_index(drop=True)
    data_test = data_test.reset_index(drop=True)

    dataset_save_path += "/" + str(train_rate) + "/"
    if not os.path.exists(dataset_save_path):
        os.mkdir(dataset_save_path)

    write_dataset_tsv(data_train['datagram'],data_train['label'], dataset_save_path, "train_"+str(samples))
    write_dataset_tsv(data_test['datagram'], data_test['label'], dataset_save_path, "test_"+str(samples))
    write_dataset_tsv(data_val['datagram'], data_val['label'], dataset_save_path, "valid_"+str(samples))

    print("finish generating pre-train's datagram dataset.\nPlease check in %s" % dataset_save_path)


