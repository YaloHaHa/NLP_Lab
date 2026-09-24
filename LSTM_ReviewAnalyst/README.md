# LSTM_ReviewAnalyst

LSTM_ReviewAnalyst is a Chinese sentiment analysis tool that classifies input text as positive or negative using an LSTM-based neural network. It converts free-form text into a practical sentiment signal for downstream analysis.

## Tool Overview
- Input: Chinese text (for example, user reviews or short comments).
- Core pipeline: text preprocessing -> tokenization -> vocabulary indexing -> LSTM inference.
- Output: sentiment label (Positive or Negative) with a confidence score.

## Architecture Diagram (draw.io)
The script relationship diagram is versioned in this repository:
- Source (.drawio): [docs/diagrams/2026.09.19_LSTM_ReviewAnalyst.drawio](docs/diagrams/2026.09.19_LSTM_ReviewAnalyst.drawio)

Recommended practice:
1. Keep the .drawio file as the editable source of truth.
2. Export a rendered SVG or PNG from draw.io for easy preview in PRs and docs.
3. Update the diagram whenever pipeline dependencies change.

Diagram legend:
- Box: script/module
- Arrow: runtime dependency or data flow

## Pipeline Flow (High-Level)
1. process.py loads raw labeled reviews, splits data, builds vocabulary, and writes processed datasets.
2. dataset.py loads processed JSONL files and yields PyTorch tensors through DataLoader.
3. train.py trains the LSTM model and saves the best checkpoint.
4. evaluate.py loads the model and reports test accuracy.
5. predict.py provides interactive inference for user-entered text.

## Script Map
- [src/process.py](src/process.py): preprocessing and dataset generation.
- [src/dataset.py](src/dataset.py): Dataset and DataLoader definitions.
- [src/tokenizer.py](src/tokenizer.py): jieba-based tokenizer and vocabulary tooling.
- [src/model.py](src/model.py): LSTM sentiment model definition.
- [src/train.py](src/train.py): training entry point.
- [src/evaluate.py](src/evaluate.py): evaluation entry point.
- [src/predict.py](src/predict.py): interactive prediction entry point.

## Example Usage (Interactive Prediction)
Example session:

```text
> 是个啥哇
Negative sentiment detected: 0.4247146546840668

> 别给我哇哇叫
Negative sentiment detected: 0.12905997037887573

> emmm我得想想
Positive sentiment detected: 0.6065571308135986
```

Interpretation:
- Scores above 0.5 are interpreted as Positive sentiment.
- Scores at or below 0.5 are interpreted as Negative sentiment.

## Test Set Performance
Final evaluation output:

```text
Vocabulary loaded successfully
Model loaded successfully
Evaluation result
accuracy: 0.9157307845479888
```

This indicates approximately 91.57% test accuracy on the held-out test set for binary sentiment classification.
