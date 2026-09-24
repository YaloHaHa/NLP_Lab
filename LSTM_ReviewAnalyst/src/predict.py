# Prediction script for the Review Analyst Model to actually put it into use for sentiment analysis of user-provided text.

import jieba
import torch
import config
from model import ReviewAnalystModel
from tokenizer import JiebaTokenizer


def predict_batch(model, inputs):
    """
    Perform batch prediction using the model.

    Args:
        model (torch.nn.Module): The trained model.
        inputs (torch.Tensor): Input tensor of shape [batch_size, seq_len].

    Returns:
        list of float: List of prediction scores for each input in the batch.
    """
    model.eval()
    with torch.no_grad():
        output = model(inputs)
        # output.shape: [batch_size]
        output = torch.sigmoid(output)
    return output.tolist()


def predict(text, model, tokenizer, device):
    # 1. Process input
    indexes = tokenizer.encode(text, seq_len=config.SEQ_LEN)
    input_tensor = torch.tensor([indexes], dtype=torch.long)
    input_tensor = input_tensor.to(device)
    # input_tensor.shape: [batch_size, seq_len]

    # 2. Prediction logic
    result = predict_batch(model, input_tensor)
    return result[0] # to return the value through result[0]. 

def run_predict():
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

    print("Welcome to the Review Analyst Model (type 'q' or 'quit' to exit)")

    while True:
        user_input = input("> ")
        if user_input in ['q', 'quit']:
            print("Thank you for using the Review Analyst Model. See you next time!")
            break
        if user_input.strip() == '':
            print("Please enter some content")
            continue

        result = predict(user_input, model, tokenizer, device)
        if result > 0.5:
            print(f'Positive sentiment detected: confidence level {result}')
        else:
            print(f'Negative sentiment detected: confidence level {1- result}')


if __name__ == '__main__':
    run_predict()
