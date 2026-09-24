# Tokenizer implementation using Jieba for Chinese text segmentation

import jieba
from tqdm import tqdm
import config


class JiebaTokenizer:
    unk_token = '<unk>'
    pad_token = '<pad>'

    def __init__(self, vocab_list):
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        self.word2index = {word: index for index, word in enumerate(vocab_list)}
        self.index2word = {index: word for index, word in enumerate(vocab_list)}
        self.unk_token_index = self.word2index[self.unk_token]
        self.pad_token_index = self.word2index[self.pad_token]

    @staticmethod
    def tokenize(text):
        return jieba.lcut(text)

    def encode(self, text, seq_len):
        """
        Encode a given text into a list of token indices.

        Args:
            text (str): The input text to encode.
        Returns:
            list of int: List of token indices corresponding to the input text.
        """

        tokens = self.tokenize(text)

        # Truncate or pad the token list to match the desired sequence length
        if len(tokens) > seq_len:
            tokens = tokens[:seq_len]
        elif len(tokens) < seq_len:
            tokens += [self.pad_token] * (seq_len - len(tokens))

        return [self.word2index.get(token, self.unk_token_index) for token in tokens]

    @classmethod
    def build_vocab(cls, sentences, vocab_path):
        """
        Build vocabulary from a list of sentences and save it to the specified path.

        Args:
            sentences (list of str): List of sentences to build the vocabulary from.
            vocab_path (str or Path): Path to save the vocabulary file.
        Returns:
            None
        """
        vocab_set = set()
        for sentence in tqdm(sentences, desc="Building vocabulary"):
            vocab_set.update(jieba.lcut(sentence))

        vocab_list = [cls.pad_token, cls.unk_token] + [token for token in vocab_set if token.strip() != '']
        print(f'Vocabulary size: {len(vocab_list)}')

        # Save vocabulary
        with open(vocab_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(vocab_list))

    @classmethod
    def from_vocab(cls, vocab_path):
        """
        Create a JiebaTokenizer instance from a saved vocabulary file.

        Args:
            vocab_path (str or Path): Path to the vocabulary file.
        Returns:
            JiebaTokenizer: An instance of JiebaTokenizer initialized with the vocabulary.
        """
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_list = [line.strip() for line in f.readlines()]
        return cls(vocab_list)


if __name__ == '__main__':
    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab.txt')
    print(f'Vocabulary size: {tokenizer.vocab_size}')
    print(f'Special token: {tokenizer.unk_token}')
    print(tokenizer.encode("今天天气不错"))
