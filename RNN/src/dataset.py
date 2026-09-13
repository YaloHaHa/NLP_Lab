# dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

import config

class InputMethodDataset(Dataset):
    """
    Input method dataset class for loading a JSONL file and creating tensors.
    """

    def __init__(self, file_path):
        """
        Initialize the dataset.

        :param file_path: Path to the data file (JSONL format).
        """
        self.data = pd.read_json(file_path, lines=True).to_dict(orient='records')

    def __len__(self):
        """
        Return the number of samples in the dataset.

        :return: Number of samples.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Get the data sample at the given index.

        :param index: Sample index.
        :return: (input_tensor, target_tensor)
        """
        input_tensor = torch.tensor(self.data[index]['input'], dtype=torch.long)
        target_tensor = torch.tensor(self.data[index]['target'], dtype=torch.long)
        return input_tensor, target_tensor

def get_dataloader(train=True):
    """
    Get the data loader.

    :param train: Whether to load the training set (True) or test set (False).
    :return: DataLoader object.
    """
    file_name = 'train_dataset.jsonl' if train else 'test_dataset.jsonl'
    dataset = InputMethodDataset(config.PROCESSED_DATA_DIR / file_name)
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

if __name__ == '__main__':
    dataloader = get_dataloader()
    for input_tensor, target_tensor in dataloader:
        print(input_tensor.shape, target_tensor.shape)
        break
