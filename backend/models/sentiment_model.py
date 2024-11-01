import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader

# Constants
MODEL_NAME = 'bert-base-uncased'
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data(file_path):
    df = pd.read_csv(file_path, encoding='ISO-8859-1', header=None)
    df.columns = ["target", "ids", "date", "flag", "user", "text"]

    sentiment_map = {0: 0, 2: 1, 4: 2}  # 0=negative, 1=neutral, 2=positive
    df['sentiment_label'] = df['target'].map(sentiment_map)
    
    # Split data into train and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'], df['sentiment_label'], test_size=0.2, random_state=42
    )
    return train_texts.tolist(), val_texts.tolist(), train_labels.tolist(), val_labels.tolist()

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

# Training function
def train_model(train_texts, train_labels, val_texts, val_labels):
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()
    return trainer

def save_model(trainer, save_path):
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)

if __name__ == "__main__":
    file_path = "../../Sentiment140/training.1600000.processed.noemoticon.csv"
    train_texts, val_texts, train_labels, val_labels = load_data(file_path)
    
    trainer = train_model(train_texts, train_labels, val_texts, val_labels)
    save_model(trainer, "sentiment_model")

    print("Model training complete and saved.")
