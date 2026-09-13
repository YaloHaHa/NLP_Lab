# Evaluate the trained RNN model on a given dataset
import torch
from tqdm import tqdm

import config
from model import InputMethodModel
from dataset import get_dataloader
from predict import predict_batch

def evaluate(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        top1_correct = 0
        top5_correct = 0
        total = 0
        for inputs, targets in tqdm(dataloader):
            inputs = inputs.to(device)
            targets = targets.tolist()
            top5_indexes_list = predict_batch(model, inputs)
            for pred, target in zip(top5_indexes_list, targets):
                if target == pred[0]:
                    top1_correct += 1
                if target in pred:
                    top5_correct += 1
                total += 1
        top1_acc = top1_correct / total if total > 0 else 0
        top5_acc = top5_correct / total if total > 0 else 0
        return top1_acc, top5_acc


def run_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(config.MODELS_DIR / "vocab.txt", "r") as f:
        vocab_list = [line.strip() for line in f.readlines()]
    word2index = {word: idx for idx, word in enumerate(vocab_list)}
    index2word = {idx: word for idx, word in enumerate(vocab_list)}
    print("Vocabulary size:", len(vocab_list))

    # Load the trained model
    model = InputMethodModel(vocab_size=len(vocab_list)).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / "best.pth", map_location=device))
    model.eval()

    # Get the dataloader for the evaluation dataset
    dataloader = get_dataloader(train=False)

    # Evaluate the model on the evaluation dataset
    top1_acc, top5_acc = evaluate(model, dataloader, device)
    print(f"Top-1 Accuracy: {top1_acc:.4f}")
    print(f"Top-5 Accuracy: {top5_acc:.4f}")

if __name__ == "__main__":
    run_evaluate()