import os
import re
import pyttsx3

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

    engine = pyttsx3.init()
    engine.setProperty('rate', 180)

    file_count = 0
    with open(CSV_METADATA, 'w', encoding='utf-8') as f:
        for root, _, files in os.walk(INPUT_DIR):
            for file in files:
                if not file.endswith('.txt'): continue

                with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as txt_f:
                    content = txt_f.read()
                    lines = re.split(r'[，。！？\n\r]', content)

                    for line in lines:
                        pure_text = get_pure_chinese(line)
                        if MIN_LEN <= len(pure_text) <= MAX_LEN:
                            audio_name = f"voc_{file_count:06d}.wav"
                            audio_path = os.path.join(OUTPUT_DIR, audio_name)

                            engine.save_to_file(pure_text, audio_path)
                            f.write(f"{audio_path}\t{pure_text}\n")
                            file_count += 1

                            if file_count % 100 == 0:
                                engine.runAndWait()
                                print(f"Progress: {file_count}")

        engine.runAndWait()

    print(f"Done. Total: {file_count}")


if __name__ == "__main__":
    prepare_raw_dataset()