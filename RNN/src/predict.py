# Predict using the trained RNN model
# Workflow: text -> tokenizer -> vocab table-> token ID -> tensor -> model forward prop -> top5 -> vocab table -> text
from model import InputMethodModel
import config
import torch
import jieba

def predict_batch(model, inputs):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = inputs.to(device)
    with torch.no_grad():
        output = model(inputs)
    top5_prob, top5_idx = torch.topk(output, 5)
    top5_indexes_list = top5_idx.tolist()
    return top5_indexes_list


def predict(text):
    """
    Predict the top 5 words for the given input text using the trained RNN model.

    Args:
        text: The input text to predict.

    Returns:
        A list of the top 5 predicted words.
    """
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Vocab list
    with open (config.MODELS_DIR / "vocab.txt", "r") as f:
        vocab_list = [line.strip() for line in f.readlines()]
    word2index = {word: idx for idx, word in enumerate (vocab_list)}
    index2word = {idx: word for idx, word in enumerate(vocab_list)}
    # Tokenize the input text
    tokens = jieba.lcut(text)
    indexes = [word2index.get(token, 0) for token in tokens]
    # Convert token IDs to tensor
    input_tensor = torch.tensor([indexes],dtype=torch.long).to(device)
    # Load the trained model
    model = InputMethodModel(vocab_size=len(vocab_list)).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / "best.pth", map_location=device))
    model.eval()  # Set the model to evaluation mode
    # Forward pass through the model
    with torch.no_grad():
        output = model(input_tensor)
    # Get the top 5 predictions
    top5_prob, top5_idx = torch.topk(output,5)
    # Convert the top 5 indices to words through vocab table
    top5_words = [index2word[idx.item()] for idx in top5_idx[0]]
    return top5_words

def run_predict():
    print("请输入文本: (输入q退出) ")
    text = ''
    while True:
        user_input = input(':')
        if user_input == 'q':
            print("退出预测")
            break
        if not user_input:
            print("输入不能为空")
            continue
        text += user_input
        print("当前输入文本:", text)
        top5_words = predict(text)
        print(top5_words)
    

if __name__ == "__main__":
    #text = '今天天气真好'
    #top5_words = predict(text)
    #print(top5_words)

    run_predict()