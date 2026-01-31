import json

def get_vocab(meta_data_dir,vocab_data_dump_dir="vocab_data.json"):
    vocab_data = {
        "tokens" : dict(),
        "total_frequency" : 0
    }
    with open(meta_data_dir, "r", encoding="utf-8") as r:
        lines = [line for line in r.readlines() if line.strip()]
        for line in lines:
            line = line.strip()
            split_units = line.split("\t")
            label_str = split_units[-1].replace(" ", "")
            labels = list(label_str)
            for label in labels:
                vocab_data["tokens"][label] = vocab_data["tokens"].get(label, 0) + 1
                vocab_data["total_frequency"] += 1
    with open(vocab_data_dump_dir, "w", encoding="utf-8") as w:
        json.dump(vocab_data, w)

def get_clean_vocab_and_meta_data(meta_data_dir,vocab_data_dump_dir="vocab_data.json",clean_meta_data_dump_dir="clean_meta_data.json",skip_threshold=10):
    vocab_data = json.load(open("vocab_data.json", "r", encoding="utf-8"))
    clean_meta_data = dict()
    with open(meta_data_dir, "r", encoding="utf-8") as r:
        lines = [line for line in r.readlines() if line.strip()]
        print(f"total count of original vocab data = {len(vocab_data["tokens"])}")
        print(f"total frequency of original vocab data = {vocab_data["total_frequency"]}")
        print(f"total count of original meta data = {len(lines)}")
        for line in lines:
            line = line.strip()
            split_units = line.split("\t")
            label_str = split_units[-1].replace(" ", "")
            labels = list(label_str)
            skip = False
            for label in labels:
                frequency = vocab_data["tokens"][label]
                if frequency < skip_threshold:
                    skip = True
                    break
            if not skip:
                audio_rel_path = split_units[0]
                clean_meta_data[audio_rel_path] = label_str

    keys = list(vocab_data["tokens"].keys())
    for k in keys:
        if vocab_data["tokens"][k] < skip_threshold:
            vocab_data["total_frequency"] -= vocab_data["tokens"][k]
            del vocab_data["tokens"][k]

    print()
    print(f"total count of new vocab data = {len(vocab_data["tokens"])}")
    print(f"total frequency of new vocab data = {vocab_data["total_frequency"]}")
    print(f"total count of new meta data = {len(clean_meta_data)}")

    with open(clean_meta_data_dump_dir, "w", encoding="utf-8") as w:
        json.dump(clean_meta_data, w)

    with open(vocab_data_dump_dir, "w", encoding="utf-8") as w:
        json.dump(vocab_data, w)

def check_vocab_data(vocab_dir="vocab_data.json"):
    check_dataset = [
        "打开头灯","头盔","安全","上报","云端","请示","危险情况",
        "摔倒","坠落","塌方","瓦斯泄露"
    ]
    check_set = set(list("".join(check_dataset)))
    with open(vocab_dir, "r", encoding="utf-8") as r:
        vocab_data = json.load(r)
        tokens = vocab_data["tokens"]
        entire_set = set(tokens.keys())
        if len(check_set & entire_set) != len(check_set):
            raise Exception("The vocabulary is too small, please add more training data")

    print("Data check passed")

if __name__ == "__main__":
    vocab_data_dump_dir = "vocab_data.json"
    meta_data_dir = r"C:\Files\TrainingDatas\zhvoice\metadata.csv"
    clean_meta_data_dump_dir = "clean_meta_data.json"
    get_vocab(meta_data_dir=meta_data_dir,
              vocab_data_dump_dir=vocab_data_dump_dir)
    get_clean_vocab_and_meta_data(meta_data_dir=meta_data_dir,
                                 vocab_data_dump_dir=vocab_data_dump_dir,
                                 clean_meta_data_dump_dir=clean_meta_data_dump_dir,
                                 skip_threshold=10)
    check_vocab_data(vocab_dir=vocab_data_dump_dir)