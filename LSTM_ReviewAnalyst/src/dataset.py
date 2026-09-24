# 1.Load processed data and define the dataset class
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import config


class ReviewAnalystDataset(Dataset):
    def __init__(self, path):
        self.data = pd.read_json(path, lines=True, orient='records').to_dict(orient='records')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_tensor = torch.tensor(self.data[index]['review'], dtype=torch.long)
        target_tensor = torch.tensor(self.data[index]['label'], dtype=torch.float)
        return input_tensor, target_tensor


# 2. Provide a function to get the dataloader for training and testing
def get_dataloader(train=True):
    path = config.PROCESSED_DATA_DIR / ('train.jsonl' if train else 'test.jsonl')
    dataset = ReviewAnalystDataset(path)
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)


if __name__ == '__main__':
    train_dataloader = get_dataloader()
    test_dataloader = get_dataloader(train=False)
    print(len(train_dataloader))
    print(len(test_dataloader))

    for input_tensor, target_tensor in train_dataloader:
        print(input_tensor.shape)  # input_tensor.shape: [batch_size, seq_len]
        print(target_tensor.shape)  # target_tensor.shape : [batch_size]
        break
