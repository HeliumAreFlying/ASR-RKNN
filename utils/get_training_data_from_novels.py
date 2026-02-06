import os
import re
import comtypes.client

INPUT_DIR = r'C:\Users\Administrator\Documents\xwechat_files\wxid_8ly89bpuixxa22_6c25\msg\file\2026-01'
OUTPUT_DIR = 'basic_data/generated_training_data/waves'
CSV_METADATA = 'basic_data/generated_training_data/metadata.csv'
MIN_LEN = 8
MAX_LEN = 32


def get_pure_chinese(text):
    return "".join(re.findall(r'[\u4e00-\u9fa5]', text))


def prepare_raw_dataset():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    tasks = []
    file_count = 0
    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            if not file.endswith('.txt'): continue
            with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as txt_f:
                lines = re.split(r'[，。！？\n\r]', txt_f.read())
                for line in lines:
                    pure_text = get_pure_chinese(line)
                    if MIN_LEN <= len(pure_text) <= MAX_LEN:
                        audio_path = os.path.abspath(os.path.join(OUTPUT_DIR, f"voc_{file_count:06d}.wav"))
                        tasks.append((audio_path, pure_text))
                        file_count += 1

    with open(CSV_METADATA, 'w', encoding='utf-8') as f:
        for audio_path, pure_text in tasks:
            f.write(f"{audio_path}{pure_text}\n")

    speak = comtypes.client.CreateObject("Sapi.SpVoice")
    filestream = comtypes.client.CreateObject("Sapi.SpFileStream")

    total = len(tasks)
    for idx, (audio_path, pure_text) in enumerate(tasks):
        filestream.Open(audio_path, 3, False)
        speak.AudioOutputStream = filestream
        speak.Speak(pure_text)
        filestream.Close()
        if (idx + 1) % 50 == 0:
            print(f"Fast Exporting: {idx + 1} / {total}")


if __name__ == "__main__":
    prepare_raw_dataset()