# AI_News_validator# 

📰 Fake Factual News Predictor

A Python-based machine learning project that predicts whether a news article is **fake** or **factual** using text data. The model is trained on a dataset (`myNewsData.csv`) containing labeled samples of news content.

## 📂 Project Structure

## 📌 Features

- Reads and processes a labeled dataset (`myNewsData.csv`)
- Cleans and vectorizes news text data
- Trains a classification model (e.g., Logistic Regression or Naive Bayes)
- Predicts whether news content is fake or factual
- Includes accuracy metrics and confusion matrix for evaluation

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/GChhabra78/AI_News_validator.git
cd AI_News_validator

pip install -r requirements.txt

python fakeNewsCaseStudy.py

The script will:

Load myData.csv

Train a model

Output performance metrics

Optionally allow you to test with new custom inputs

🧪 Sample Code Snippet
python
Copy
Edit
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv("myData.csv")

# Preprocessing
X = df["text"]
y = df["label"]
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)
print("Accuracy:", model.score(X_test, y_test))

📊 Dataset Format (myData.csv)
Your dataset should be structured like this:

title,text,date,fake_or_factual
"First Title","Breaking: New virus discovered","April 6, 2017",Fake News
"Second Title","President signs new climate bill","May 8, 2023",Factual News

🤝 Contributing
Contributions are welcome! To contribute:

Fork the repo

Create a branch (git checkout -b feature-name)

Make changes

Commit (git commit -m 'Add feature')

Push (git push origin feature-name)

Open a Pull Request

📄 License
This project is licensed under the MIT License. See the LICENSE file for more details.

📧 Contact
Have questions? Reach out via chhabragitika@yahoo.com
