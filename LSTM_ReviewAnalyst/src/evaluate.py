# Evaluation script for the ReviewAnalystModel

import torch
import config
from model import ReviewAnalystModel
from dataset import get_dataloader
from predict import predict_batch
from tokenizer import JiebaTokenizer


def evaluate(model, test_dataloader, device):
    total_count = 0
    correct_count = 0

    for inputs, targets in test_dataloader:
        inputs = inputs.to(device)
        # inputs.shape: [batch_size, seq_len]
        targets = targets.tolist()
        # targets.shape: [batch_size] e.g.[1,0,1]
        output = predict_batch(model, inputs)
        # output.shape: [batch_size] e.g. [0.9, 0.1, 0.8]
        for result, target in zip(output, targets):
            total_count += 1
            result = 1 if result > 0.5 else 0
            if target == result:
                correct_count += 1
    return correct_count / total_count if total_count > 0 else 0.0


def run_evaluate():
    # Prepare resources
    # 1. Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. Vocabulary
    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab.txt')
    print("Vocabulary loaded successfully")

    # 3. Model
    model = ReviewAnalystModel(vocab_size=tokenizer.vocab_size, padding_index=tokenizer.pad_token_index).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'best_model.pth'))
    print("Model loaded successfully")

    # 4. Dataset
    test_dataloader = get_dataloader(train=False)

    # 5. Evaluation logic
    acc = evaluate(model, test_dataloader, device)
    print("Evaluation result")
    print(f"accuracy: {acc}")


if __name__ == '__main__':
    run_evaluate()
