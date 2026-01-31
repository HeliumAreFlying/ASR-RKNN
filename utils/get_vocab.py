import json

def get_vocab(meta_data_dir):
    vocab_data = {
        "tokens" : dict(),
        "vocab_size" : 0
    }
    with open(meta_data_dir, "r", encoding="utf-8") as r:
        lines = r.readlines()
        for line in lines:
            line = line.strip()
            labels = list(line.split("\t")[-1])
            for label in labels:
                vocab_data["tokens"][label] = vocab_data["tokens"].get(label, 0) + 1
                vocab_data["vocab_size"] += 1
    with open("vocab_data.json", "w", encoding="utf-8") as w:
        json.dump(vocab_data, w)

if __name__ == "__main__":
    get_vocab(meta_data_dir=r"C:\Files\TrainingDatas\zhvoice\metadata.csv")