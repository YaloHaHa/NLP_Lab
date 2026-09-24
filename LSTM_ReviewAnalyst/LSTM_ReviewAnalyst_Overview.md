# LSTM_ReviewAnalyst One-Pager

## 1. Executive Summary
LSTM_ReviewAnalyst is a Chinese sentiment analysis tool that classifies input text as positive or negative using an LSTM-based neural network. It is designed to turn free-form review text into a quick, actionable sentiment signal for downstream analysis.

## 2. Tool Overview
- **Input:** Chinese text (for example, user reviews or short comments).
- **Core pipeline:** text preprocessing -> tokenization -> vocabulary indexing -> LSTM inference.
- **Output:** sentiment label (**Positive** or **Negative**) with a confidence score.

## 3. How It Works (High-Level)
1. **Data processing**
   - Load raw labeled review data.
   - Split into training and testing sets.
2. **Vocabulary + encoding**
   - Build a vocabulary from training text.
   - Convert each sentence into fixed-length token index sequences.
3. **Model training**
   - Train an LSTM sentiment classifier.
   - Save the best model checkpoint.
4. **Inference and evaluation**
   - Use the saved model for interactive predictions.
   - Run test-set evaluation to report final accuracy.

## 4. Example Usage (Interactive Prediction)
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
- Scores above `0.5` are interpreted as **Positive sentiment**.
- Scores at or below `0.5` are interpreted as **Negative sentiment**.

## 5. Test Set Performance
Final evaluation output:

```text
Vocabulary loaded successfully
Model loaded successfully
Evaluation result
accuracy: 0.9157307845479888
```

This indicates approximately **91.57% test accuracy**, showing strong overall performance on the held-out test set for binary sentiment classification.
