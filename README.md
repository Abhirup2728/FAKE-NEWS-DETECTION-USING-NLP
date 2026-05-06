# 📰 Fake News Detection Using NLP

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red?logo=streamlit)](https://fake-news-detector-app-brainware.streamlit.app/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-98.71%25-brightgreen)]()


---

## 🚀 Live Demo

🌐 **[https://fake-news-detector-app-brainware.streamlit.app/](https://fake-news-detector-app-brainware.streamlit.app/)**

Paste any news article or headline and get an instant **REAL / FAKE** prediction with confidence probability.

---

## 👥 Team

| Name | Student Code | Roll No. | Reg. No. |
|------|-------------|----------|----------|
| Debojyoti Deb | BWU/BTA/22/590 | 22010332556 | 22013001890 |
| Abhirup Gumtya | BWU/BTA/22/557 | 22010332523 | 22013001857 |
| Pralay Bajkhan | BWU/BTA/22/567 | 22010332533 | 22013001867 |
| Ankita Dhali | BWU/BTA/22/613 | 22010332579 | 22013001913 |

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Results](#-results)
- [Deployment](#-deployment)
- [Future Scope](#-future-scope)
- [References](#-references)

---

## 🧠 Overview

With the explosive growth of social media, fake news has become one of the most critical challenges of our time. Misinformation affects elections, public health, financial markets, and societal trust.

This project builds an **end-to-end automated fake news detection system** using **Natural Language Processing (NLP)** and **Machine Learning**. The pipeline covers:

- Raw text ingestion and cleaning
- TF-IDF feature extraction
- Multi-model training and evaluation
- Live deployment as a Streamlit web application

The final deployed model (Logistic Regression + TF-IDF) achieves **98.71% accuracy** on a held-out test set of 8,980 samples.

---

## 📊 Dataset

**Source:** [Kaggle Fake News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

| File | Label | Count |
|------|-------|-------|
| `True.csv` | Real News (1) | 21,417 |
| `Fake.csv` | Fake News (0) | 23,481 |
| **Combined** | — | **44,898** |

- **Train / Test Split:** 80% / 20% (stratified)
- **Class Balance:** 52.3% Fake / 47.7% Real — near-balanced, no SMOTE needed

Each article contains: `title`, `text`, `subject`, `date`. Title and body text were concatenated into a single `full_text` feature for maximum signal.

---

## 🏗️ System Architecture

```
Raw News Text
     │
     ▼
┌─────────────────────────┐
│   Text Preprocessing    │  ← Regex cleaning, lowercasing,
│                         │    tokenization, stopword removal
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   TF-IDF Vectorization  │  ← max_df=0.7, fit on train only
│   (max_df = 0.7)        │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   Model Training        │  ← Logistic Regression (primary)
│   & Evaluation          │    + DT, RF, PAC, NB (comparison)
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Streamlit Deployment   │  ← Real-time user prediction
│  (Community Cloud)      │
└─────────────────────────┘
```

---

## 🛠️ Tech Stack

| Category | Tools / Libraries |
|----------|------------------|
| Language | Python 3.10 |
| ML Framework | Scikit-learn |
| NLP | NLTK (tokenization, stopwords) |
| Feature Extraction | TF-IDF Vectorizer |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Serialization | Joblib |
| Deployment | Streamlit Community Cloud |
| Environment | Google Colab + Local Python |

---

## 📁 Project Structure

```
fake-news-detection/
│
├── dataset/
│   ├── True.csv                    # Real news articles
│   └── Fake.csv                    # Fake news articles
│
├── notebooks/
│   └── fake_news_detection.ipynb   # Full EDA + training notebook
│
├── app/
│   ├── app.py                      # Streamlit web application
│   ├── final_model.pkl             # Serialized Logistic Regression model
│   └── final_vectorizer.pkl        # Serialized TF-IDF vectorizer
│
├── reports/
│   └── FakeNewsDetection_ProjectReport_BWU.pdf
│
├── images/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── accuracy_comparison.png
│   └── streamlit_screenshot.png
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/fake-news-detection.git
cd fake-news-detection
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
pandas
numpy
scikit-learn
nltk
matplotlib
seaborn
joblib
streamlit
```

### 4. Download NLTK Data

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### 5. Add Dataset

Download `True.csv` and `Fake.csv` from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) and place them in the `dataset/` folder.

---

## 💻 Usage

### Run the Jupyter Notebook

```bash
jupyter notebook notebooks/fake_news_detection.ipynb
```

### Run the Streamlit App Locally

```bash
streamlit run app/app.py
```

Then open `http://localhost:8501` in your browser.

### Predict Programmatically

```python
import joblib
import nltk
from nltk.corpus import stopwords

# Load model and vectorizer
model = joblib.load('app/final_model.pkl')
vectorizer = joblib.load('app/final_vectorizer.pkl')

stop_words = set(stopwords.words('english'))

def predict(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    vector = vectorizer.transform([" ".join(tokens)])
    prediction = model.predict(vector)[0]
    confidence = model.predict_proba(vector).max()
    return "REAL" if prediction == 1 else "FAKE", round(confidence * 100, 2)

label, confidence = predict("Your news article text here...")
print(f"Prediction: {label} ({confidence}% confidence)")
```

---

## 📈 Model Performance

### Primary Model: Logistic Regression (TF-IDF)

| Metric | Real (Class 0) | Fake (Class 1) | Overall |
|--------|---------------|---------------|---------|
| Precision | 0.98 | 0.99 | 0.99 |
| Recall | 0.99 | 0.98 | 0.99 |
| F1-Score | 0.99 | 0.99 | 0.99 |
| **Accuracy** | — | — | **98.71%** |
| **AUC-ROC** | — | — | **> 0.98** |

### Confusion Matrix

```
                 Predicted
                 Fake    Real
Actual  Fake  [ 4,596    100 ]
        Real  [    43  4,241 ]
```

✅ Near-symmetric errors — no class bias detected.

### Multi-Model Comparison (same dataset)

| Model | Accuracy |
|-------|---------|
| **Logistic Regression** ⭐ | **98.71%** |
| Passive Aggressive Classifier | 99.69% |
| Random Forest | 99.80% |
| Decision Tree | 99.70% |
| Multinomial Naive Bayes | 94.00% |

> **Note:** While tree-based models show near-perfect accuracy on this specific Kaggle dataset (suggesting possible overfitting due to source-domain leakage), the **Logistic Regression model generalizes better** on truly unseen news from external sources and was therefore chosen for deployment.

---

## 🌐 Deployment

The trained model is deployed on **Streamlit Community Cloud**.

**Live URL:** [https://fake-news-detector-app-brainware.streamlit.app/](https://fake-news-detector-app-brainware.streamlit.app/)

The app:
1. Accepts raw news text input from the user
2. Applies the same preprocessing pipeline used during training
3. Vectorizes text using the serialized TF-IDF vectorizer
4. Returns a **REAL / FAKE** label with confidence probability

---

## 🔭 Future Scope

1. **Deep Learning** — Integrate LSTM, Bi-LSTM, BERT, or RoBERTa for richer contextual understanding (target accuracy: >99%)
2. **Multilingual Support** — Extend to Bengali, Hindi, Tamil using mBERT / XLM-RoBERTa
3. **Social Media Integration** — Auto-fetch trending posts via Twitter, Reddit, Telegram APIs
4. **Deepfake Detection** — Incorporate computer vision models for multimedia misinformation
5. **Browser Extension** — Real-time credibility overlay on news websites
6. **Mobile App** — Cross-platform Android/iOS news verification tool
7. **Continual Learning** — Periodic retraining on fresh labelled data to adapt to evolving fake news patterns
8. **Source Credibility Scoring** — Domain reputation scores alongside article-level predictions

---

## 📚 References

1. Gupta, S., & Jain, R. (2023). *Application of AI Techniques to Detect Fake News.* Electronics, 12(24), 5041. MDPI.
2. Alshuwaier, F. A., & Alsulaiman, M. (2024). *Fake News Detection Using ML and DL Algorithms.* Computers, 14(9), 394. MDPI.
3. Ajao, O., & Bhowmik, D. (2021). *Detecting Fake News Using Machine Learning: A Systematic Literature Review.* arXiv:2102.04458.
4. Zhou, X., & Zafarani, R. (2018). *A Survey on NLP for Fake News Detection.* arXiv:1811.00770.
5. Devlin, J. et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers.* arXiv:1810.04805.
6. Wang, W. Y. (2017). *"Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection.* ACL 2017.
7. [Scikit-learn Documentation](https://scikit-learn.org)
8. [NLTK Documentation](https://www.nltk.org)
9. [Streamlit Documentation](https://docs.streamlit.io)
10. [Kaggle Fake News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">Made with ❤️ by Team Brainware | CSE (AI & ML) Batch 2022–2026</p>
