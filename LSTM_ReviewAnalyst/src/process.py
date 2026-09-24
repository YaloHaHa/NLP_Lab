import pandas as pd
from sklearn.model_selection import train_test_split
import config
from tokenizer import JiebaTokenizer

def process():
    print("Processing data...")
    # Read raw file & Process raw data
    df = pd.read_csv(config.RAW_DATA_DIR/'online_shopping_10_cats.csv',usecols=['label','review'],
                     encoding='utf-8').dropna()

    # Split the data into training and testing sets
        # To have the same distribution of classes in both sets, use stratify=df['label'] in train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

    # Build vocabulary from the training data
        # Special tokens like <PAD>, <UNK>
    JiebaTokenizer.build_vocab(train_df['review'].tolist(), config.MODELS_DIR / 'vocab.txt')

    # Build datasets for training and testing and save them
    train_df['review'] = train_df['review'].apply(lambda x: JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab.txt').encode(x, config.SEQ_LEN))
    test_df['review'] = test_df['review'].apply(lambda x: JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab.txt').encode(x, config.SEQ_LEN))
    train_df.to_json(config.PROCESSED_DATA_DIR/'train.jsonl', index=False, lines=True, orient='records', force_ascii=False)
    test_df.to_json(config.PROCESSED_DATA_DIR/'test.jsonl', index=False, lines=True, orient='records', force_ascii=False)

    print("Processing complete. Processed data saved.")

if __name__ == "__main__":
    process()