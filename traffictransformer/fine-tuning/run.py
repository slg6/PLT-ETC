import os
import subprocess
# os.chdir(os.path.expanduser("~/llmfortraffic/looptransformer/traffictransformer"))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
dataset_names = os.listdir(f"fine-data/fine_data_flow/")

# dataset_names = ['cic-iot_300s', 'cic','andro_300s', 'idi', 'isd_30_1s', 'service', 'uav_5s', 'ustcMal', 'ustc', 'app', 'tor']     #running crossterm mutilhead
# dataset_names = ["cipherspectrum_aes128", "cipherspectrum_aes256", "cipherspectrum_cha", "cipherspectrum_mix", "cstent"]     #running crossterm mutilhead
# dataset_names = ['idi','service', 'uav_5s', 'ustcMal', 'app', 'tor', 'cic-iot_300s', 'andro_300s', 'ustc']     #running crossterm mutilhead

#idi service

dataset_names = ["cipherspectrum_aes256", "cipherspectrum_cha", "cipherspectrum_mix", "cstent", 'tor', 'uav_5s', 'ustc',
                 'cic-iot_300s', 'cic','andro_300s', 'idi', 'isd_30_1s', 'service', 'ustcMal',  'app']     #running crossterm mutilhead

dataset_names = ['itcnet']
enhances = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# pretrain_steps = ['0', '1000', '2000', '3000', '4000', '5000', '6000', '7000', '8000', '9000', '10000', '11000']
pretrain_steps = ['12000']

sample = 500
tsv_path_template = "fine-data/fine_data_flow/{}/{}/"
enhance_factor = 1
func = "nofeature"
train_rate = 0.8

command_template = [
    "python3", f"fine-tuning/run_classifier_{func}.py",
    "--vocab_path", "vocab_gen/vocab_packet_4chars_all.txt",
    "--train_path", None,           #f"{tsv_path}train_enhanced{str(sample)}_{str(enhance_factor)}_dataset.tsv"
    "--dev_path", None,             #f"{tsv_path}valid_{str(sample)}_dataset.tsv"
    "--test_path", None,            #f"{tsv_path}test_{str(sample)}_dataset.tsv"
    "--samples", f"{str(sample)}",
    # "--pretrained_model_path", "pre-training/pre-training-model/model_packet_dataset0_test.bin-12000",
    "--pretrained_model_path", "pre-training/pre-training-model/model_packet_dataset0_test.bin-{}",
    "--epochs_num", "100",
    "--earlystop", "10",
    "--batch_size", "32",
    "--embedding", "word_pos_seg",
    "--encoder", "transformer",
    "--mask", "fully_visible",
    "--seq_length", "401",
    "--learning_rate", "6e-5",
    "--report_steps", "1000",
    "--config_path", "models/bert/base_config_packets.json",
]

# for dataset_name in dataset_names:
for i in range(len(dataset_names)):
    dataset_name = dataset_names[i]
    enhance = enhances[i]

    tsv_path = tsv_path_template.format(dataset_name + f"_enhance{enhance}", train_rate)

    command = command_template.copy()
    command[5] = f"{tsv_path}train_500_dataset.tsv"
    command[7] = f"{tsv_path}valid_500_dataset.tsv"
    command[9] = f"{tsv_path}test_500_dataset.tsv"
    # command[11] = f"{str(sample)}"

    print(f"------{dataset_name}")

    for step in range(len(pretrain_steps)):
        command[13] = command[13].format(pretrain_steps[step])

        result = subprocess.run(command, capture_output=True, text=True)
        print(f"STDOUT for {dataset_name}:", result.stdout)
        print(f"STDERR for {dataset_name}:", result.stderr)