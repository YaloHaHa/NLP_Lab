# Training Model

from csv import writer

import torch
from tqdm import tqdm
from dataset import get_dataloader
from model import InputMethodModel
import config
from torch.utils.tensorboard import SummaryWriter


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    """
    Train the model for one epoch.
    Args:
        model (torch.nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): DataLoader for the training data.
        loss_fn (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device to run the training on.
    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    total_loss = 0
    for inputs, targets in tqdm(dataloader, desc='training'):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad() # Clear all gradients stored on all parameters before computing gradients for the current batch
        # forward propagation
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        #backward propagation
        loss.backward() # compute gradients for each parameter
        optimizer.step() # update each parameter based on the gradient and the optimizer rule
        total_loss += loss.item()
    return total_loss / len(dataloader)

def train():
    """Train the RNN model.
    """
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Get dataset
    dataloader = get_dataloader()
    # Get the vocabulary list
    with open (config.MODELS_DIR / "vocab.txt", "r") as f:
        vocab_list = [line.strip() for line in f.readlines()]
    # Determine model
    model = InputMethodModel(vocab_size=len(vocab_list)).to(device)

    writer = SummaryWriter()
    # Determine loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    # Determine Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    # Start training loop
    best_loss = float('inf')
    for epoch in range(config.EPOCHS):
        avg_loss = train_one_epoch(model, dataloader, loss_fn, optimizer, device)
        print(f"Loss: {avg_loss:.4f}, Best Loss: {best_loss:.4f}")
        # Write to tensorboard
        writer.add_scalar('Loss/train', avg_loss, epoch)
        if avg_loss < best_loss:
            best_loss = avg_loss
            # save the best model
            torch.save(model.state_dict(), config.MODELS_DIR/"best.pth")
            print(f"New best model saved with loss: {best_loss:.4f}")
    writer.close()

if __name__ == "__main__":
    train()