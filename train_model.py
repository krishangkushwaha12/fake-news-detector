import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("fake_news.csv")

# Combine title and text
df['content'] = df['title'].fillna('') + " " + df['text'].fillna('')

# Convert labels: FAKE -> 1, REAL -> 0
df['label'] = df['label'].map({'FAKE': 1, 'REAL': 0})

# Clean text
def clean_text(text):
    text = re.sub(r"http\S+", "", str(text))
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text.lower()

df['content'] = df['content'].apply(clean_text)

X = df['content']
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression(max_iter=1000, class_weight="balanced")

model.fit(X_train_vec, y_train)

# Evaluation
pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, pred))

# Save model and vectorizer
pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))
