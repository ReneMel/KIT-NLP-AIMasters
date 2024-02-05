import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('C:/Users/renemel/Documents/KIT/Code/NLPModel/CombinedData22012024.csv')

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def tokenize_emails(emails, max_length=512):
    concatenated_emails = ' '.join(str(email) for email in emails if isinstance(email, str))
    tokenized_emails = tokenizer(concatenated_emails, max_length=max_length, truncation=True, padding=True, return_tensors='pt', add_special_tokens=True)
    return tokenized_emails

train_encodings = tokenize_emails(list(train_data['text']))
test_encodings = tokenize_emails(list(test_data['text']))

train_labels = [1 if label == 'spam' else 0 for label in list(train_data['type'])]
test_labels = [1 if label == 'spam' else 0 for label in list(test_data['type'])]

class EmailDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx].squeeze() for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item


train_dataset = EmailDataset(train_encodings, train_labels)
test_dataset = EmailDataset(test_encodings, test_labels)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
optimizer = AdamW(model.parameters(), lr=5e-5)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print("Longitud del conjunto de entrenamiento:", len(train_dataset))

for epoch in range(3):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        print("Índices utilizados:", batch_idx * train_loader.batch_size, "a", (batch_idx + 1) * train_loader.batch_size - 1)
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(true_labels, predictions)
print(f'Accuracy: {accuracy:.2f}')

classification_report_output = classification_report(true_labels, predictions)
print(f'Classification Report:\n{classification_report_output}')
