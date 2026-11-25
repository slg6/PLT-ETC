import os
# os.chdir(os.path.expanduser("~/llmfortraffic/looptransformer/traffictransformer"))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.append(os.getcwd())
from finetuning_data_gen import (
    generation_multiP,
)
def main(dataset):
    pcap_path = os.path.expanduser(f"~/llmfortraffic/dataPcap_flow_/{dataset}")

    # print("Step 1: Converting and splitting pcap files...")
    # convert_splitcap(
    #     pcapng_path=pcapng_path,
    #     is_pcap_label=True
    # )
    '''
    malware 477
    benign 3577
    all flows:  4054 2
    '''


    max_packets = 20
    enhance_factor = 1
    samples = 1000
    min_flow = 1
    traceMaxFlow = 1000
    print("Step 2: get packet data json...")
    generation_multiP(
        dataset_path=pcap_path,
        samples =samples,
        enhance_factor=enhance_factor,
        max_packets=max_packets,
        min_flow=min_flow,
        traceMaxFlow=traceMaxFlow
    )

    # print("\nStep 3: Converting to TSV format...")
    # features = ["datagram", "length", "time", "direction", "message_type"]
    # dataset_extract(
    #     dataset_save_path=dataset_save_path,
    #     features=features
    # )
    # #                           1      [SEP] 4500 ...

    # print("\nStep 4: Performing data augmentation...")
    # enhance_based_tsv(
    #     path=dataset_save_path + "dataset/",
    #     filename="train_dataset.tsv",
    #     new_file_prefix="train_enhanced",
    # )

if __name__ == "__main__":

    dataset_list = ["cipherspectrum_aes128", "cipherspectrum_aes256",
                 "cipherspectrum_cha", "cipherspectrum_mix", "cstent", "Edge-IIoTset",'ustc']
    for dataset in dataset_list:
        print(f"----------{dataset}----------")
        main(dataset)
    '''
    1. pcapng_path = "../../dataPcap/isd_5s"          # 原始pcapng文件路径
    2. max_packets = 20    #每个流最多保留的包数量
    3. min_flow = 5 #每个流最少含包的数量
    '''