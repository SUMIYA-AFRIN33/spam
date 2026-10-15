# 📱 SMS Spam Classification using SVM

A machine learning project that classifies SMS messages as **spam** or **ham (legitimate)** using a Support Vector Machine (SVM) with TF-IDF text features.

---

## 📂 Repository Contents

| File | Description |
|------|-------------|
| `SMS_Spam_SVM.ipynb` | Core notebook — SVM classifier pipeline |
| `SMS_Spam_SVM_Improved.ipynb` | Enhanced notebook — multi-model comparison + full analysis |
| `spam.csv` | Dataset (UCI SMS Spam Collection) |
| `README.md` | This file |

---

## 📊 Dataset

**Source:** [UCI SMS Spam Collection — Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset?resource=download)

| Property | Value |
|----------|-------|
| Total messages | 5,572 |
| Ham (legitimate) | 4,825 (87%) |
| Spam | 747 (13%) |
| Duplicates removed | 403 |
| Features | Raw SMS text |

---

## 🧠 Notebooks Overview

### `SMS_Spam_SVM.ipynb` — Core Notebook

A clean, end-to-end pipeline using a single SVM classifier.

**Steps:**
1. **Install & Import** — `scikit-learn`, `nltk`, `wordcloud`, `seaborn`
2. **Load Dataset** — Upload `spam.csv` from Kaggle (Google Colab)
3. **Exploratory Data Analysis (EDA)**
   - Class distribution bar chart
   - Word count histogram (spam vs ham)
   - Word clouds for spam and ham messages
4. **Text Preprocessing**
   - Lowercase conversion
   - Stopword removal
   - Lemmatization (WordNet)
5. **Train / Test Split** — 80/20 stratified split
6. **SVM Model** — `LinearSVC` inside a `Pipeline` with `TfidfVectorizer`
   - `max_features=5000`, `ngram_range=(1,2)`
7. **Hyperparameter Tuning** — `GridSearchCV` over regularization parameter `C ∈ {0.1, 1, 10}`
8. **Cross-Validation** — 5-fold CV on the best estimator
9. **Prediction Function** — Live inference on custom SMS text

---

### `SMS_Spam_SVM_Improved.ipynb` — Enhanced Pipeline

A professional-grade extension with multi-model benchmarking and richer analysis.

**Additional Steps:**

| Step | Description |
|------|-------------|
| Multi-model training | LinearSVC, Naïve Bayes, Logistic Regression |
| Model comparison table | Accuracy, Precision, Recall, F1 |
| Confusion matrices | Per-model visual |
| ROC curves | AUC comparison |
| Hyperparameter tuning | Grid search over `C` and `ngram_range` |
| Error analysis | False positives & false negatives inspection |
| Feature importance | Top discriminative TF-IDF tokens |
| Live demo | Interactive prediction on custom messages |
| Summary | Key findings and takeaways |

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading & manipulation |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | Visualizations |
| `wordcloud` | Word cloud generation |
| `nltk` | Stopwords, lemmatization |
| `scikit-learn` | ML pipeline, TF-IDF, SVM, metrics |

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install scikit-learn nltk wordcloud seaborn matplotlib pandas numpy
```

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

### 2. Get the dataset

Download `spam.csv` from [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset?resource=download) and place it in the project directory.

> **Note:** The core notebook uses Google Colab's file upload widget. If running locally, replace:
> ```python
> from google.colab import files
> uploaded = files.upload()
> ```
> with:
> ```python
> df = pd.read_csv("spam.csv", encoding="latin-1")
> ```

### 3. Run the notebook

Open either notebook in Jupyter or Google Colab and run all cells.

---

## 📈 Results

The `LinearSVC` model achieves strong performance on this dataset:

| Metric | Score |
|--------|-------|
| Accuracy | ~98–99% |
| Precision (spam) | ~97–99% |
| Recall (spam) | ~95–98% |
| F1-score (spam) | ~96–98% |

> Exact values may vary slightly based on random state and CV fold.

---

## 🔍 Key Insights

- **Spam messages** are significantly longer, use more uppercase letters, and contain more punctuation than ham.
- **TF-IDF bigrams** (`ngram_range=(1,2)`) capture context better than unigrams alone.
- **LinearSVC** is fast and highly effective for high-dimensional, sparse text data.
- **Class imbalance** (87% ham) is handled via stratified splitting.

---

## 📝 Author

**Sumaiya Afrin**

---

## 📄 License

This project is for educational purposes. The dataset is from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection).
