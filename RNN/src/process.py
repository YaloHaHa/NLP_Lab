import jieba
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import config

# An universal function for training and testing dataset preparation, which tokenizes the sentences and converts them into a suitable format for model training.
def build_dataset(sentences, word2index):
    indexed_sentences = [[word2index.get(word, 0) for word in jieba.lcut(sentence)] for sentence in sentences]
    dataset = []
    for sentence in tqdm(indexed_sentences, desc="Building dataset"):
        for i in range(len(sentence) - config.SEQ_LEN):
            input = sentence[i:i + config.SEQ_LEN]
            target = sentence[i + config.SEQ_LEN]
            dataset.append({'input': input, 'target': target})
    return dataset


def process():
    print("Data Processing Started")
    # 1.Read files
        # use .sample(frac=0.01) to randomly sample 1% of the data for the developing purpose
    df = pd.read_json(config.RAW_DATA_DIR / "synthesized_.jsonl", lines=True, orient="records").sample(frac=0.01)

    # 2.Extract sentences
    sentences = []
    for dialog in df['dialog']:
        for sentence in dialog:
            sentences.append(sentence.split("：")[1])

    # 3.Split dataset
    train_sentences, test_sentences = train_test_split(sentences, test_size=0.2)

    # 4.Build vocabulary
        #jieba.lcut is used to tokenize sentences into words
    vocab_set = set()
    for sentence in train_sentences:
        for word in jieba.lcut(sentence):
            vocab_set.add(word)
    vocab_list = ['<unk>'] + list(vocab_set)
    with open(config.MODELS_DIR / "vocab.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(vocab_list))

    # 6.Build training set
    word2index = {word: index for index, word in enumerate(vocab_list)}
    train_dataset = build_dataset(train_sentences, word2index)

    # 7.Save training set
    pd.DataFrame(train_dataset).to_json(config.PROCESSED_DATA_DIR / "train_dataset.jsonl", orient="records", lines=True)


    # 8.Build test set
    test_dataset = build_dataset(test_sentences, word2index)


    # 9.Save test set
    pd.DataFrame(test_dataset).to_json(config.PROCESSED_DATA_DIR / "test_dataset.jsonl", orient="records", lines=True)

    print("Data Processing Finished")


if __name__ == '__main__':
    process()