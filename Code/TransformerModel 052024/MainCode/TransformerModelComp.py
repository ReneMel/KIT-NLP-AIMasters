import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from torch.optim import AdamW

# Basic Dataset creation and split
file = '../Data/combined-dataset.csv'
df = pd.read_csv(file)
le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])
train_size = 0.7
train_df, test_df, train_labels, test_labels = train_test_split(
    df['text'], df['type'], test_size=train_size, random_state=42
)

class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=100):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = int(self.labels.iloc[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

model_names = [
    'distilbert-base-uncased',
    'bert-base-uncased',
    'xlnet-base-cased',
    'roberta-base',
    'albert-base-v2'
]

results = {}

for model_name in model_names:
    print(f"Training and evaluating model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = EmailDataset(train_df, train_labels, tokenizer)
    test_dataset = EmailDataset(test_df, test_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(le.classes_))
    optimizer = AdamW(model.parameters(), lr=2e-5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    epochs = 3

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        average_loss = total_loss / len(train_loader)
        print(f'Training Loss: {average_loss}')

    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    report = classification_report(all_labels, all_predictions, digits=4)
    print(report)
    results[model_name] = report

# Optional: Save the results to a file
with open('model_comparison_results.txt', 'w') as f:
    for model_name, report in results.items():
        f.write(f"Model: {model_name}\n")
        f.write(report)
        f.write("\n\n")
