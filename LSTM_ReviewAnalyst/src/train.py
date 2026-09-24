# Training script for the LSTM-based review analysis model
import time
from torch.utils.tensorboard import SummaryWriter

import torch
from dataset import get_dataloader
from tokenizer import JiebaTokenizer
import config
from model import ReviewAnalystModel

def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    """
    Train the model for one epoch.

    Args:
        model: The model to train.
        dataloader: DataLoader providing the training data.
        loss_fn: Loss function.
        optimizer: Optimizer.
        device: Device to run the training on.

    Returns:
        Average loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        # Forward pass through the model
        output = model(inputs)
        loss = loss_fn(output, targets)
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(dataloader)



def train():

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    dataloader = get_dataloader()

    # Tokenizer
    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR/ "vocab.txt")

    # Model
    model = ReviewAnalystModel(tokenizer.vocab_size, tokenizer.pad_token_index).to(device)

    # Loss function
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # TensorBoard Writer
    writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime("%Y%m%d-%H%M%S"))

    best_loss = float("inf")
    for epoch in range(config.EPOCHS):
        print(f"==== Epoch {epoch + 1} ====")
        loss = train_one_epoch(model, dataloader, loss_fn, optimizer, device)

        print(f"Epoch {epoch + 1} Loss: {loss}")
        # Log the loss to TensorBoard
        writer.add_scalar("Loss", loss, epoch)

        # Update best loss
        if loss < best_loss:
            best_loss = loss
            # Save the best model
            torch.save(model.state_dict(), config.MODELS_DIR / "best_model.pth")
            print(f"Best model saved with loss: {best_loss}")

    writer.close()

if __name__ == "__main__":
    train()