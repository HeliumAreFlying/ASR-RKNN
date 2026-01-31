import json

def get_vocab(meta_data_dir,vocab_data_dump_dir="vocab_data.json"):
    vocab_data = {
        "tokens" : dict(),
        "total_frequency" : 0
    }
    with open(meta_data_dir, "r", encoding="utf-8") as r:
        lines = r.readlines()
        for line in lines:
            line = line.strip()
            label_str = line.split("\t")[-1].replace(" ","")
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
        lines = r.readlines()
        for line in lines:
            line = line.strip()
            split_units = line.split("\t")[-1]
            label_str = split_units[-1].replace(" ", "")
            labels = list(label_str)
            skip = False
            for label in labels:
                frequency = vocab_data["tokens"][label]
                if frequency < skip_threshold:
                    skip = True
                    break
            if not skip:
                clean_meta_data[label_str] = split_units[0]

    keys = list(vocab_data["tokens"].keys())
    for k in keys:
        if vocab_data["tokens"][k] < skip_threshold:
            del vocab_data["tokens"][k]

    with open(clean_meta_data_dump_dir, "w", encoding="utf-8") as w:
        json.dump(clean_meta_data, w)

    with open(vocab_data_dump_dir, "w", encoding="utf-8") as w:
        json.dump(vocab_data, w)

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