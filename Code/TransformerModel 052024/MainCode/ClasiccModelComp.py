import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Basic Dataset creation and split
file = '../Data/combined-dataset.csv'
df = pd.read_csv(file)

# Remove or fill NaN values
df.dropna(subset=['text'], inplace=True)

# Encode labels
le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])

# Split the dataset
train_size = 0.7
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'], df['type'], test_size=train_size, random_state=42
)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=10000)
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

# Define classic models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'Support Vector Machine': SVC(kernel='linear'),
    'Naive Bayes': MultinomialNB()
}

# Train and evaluate each model
results = {}

for model_name, model in models.items():
    print(f"Training and evaluating model: {model_name}")
    model.fit(X_train, train_labels)
    predictions = model.predict(X_test)
    report = classification_report(test_labels, predictions, digits=4)
    print(report)
    results[model_name] = report

# Optional: Save the results to a file
with open('classic_model_comparison_results.txt', 'w') as f:
    for model_name, report in results.items():
        f.write(f"Model: {model_name}\n")
        f.write(report)
        f.write("\n\n")
