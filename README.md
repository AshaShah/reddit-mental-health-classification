# Multi-Class Mental Health Condition Detection in Reddit Posts

CS590 Capstone Project | Spring 2026
**Author:** Asha Shah

This project investigates whether linguistic and sentiment features extracted from Reddit posts can reliably classify text into one of five mental health conditions: **Stress, Depression, Bipolar disorder, Personality disorder, and Anxiety**. Three models of increasing representational power are compared, and SHAP explainability is used to verify that learned associations are clinically plausible.

## Results

| Model | Evaluation | Macro F1 | Accuracy |
|---|---|---|---|
| TF-IDF + Logistic Regression | 5-fold CV (n = 5,607) | **0.8125** | 0.81 |
| TF-IDF + Random Forest | 5-fold CV (n = 5,607) | **0.8234** | 0.82 |
| DistilBERT (fine-tuned) | 80/20 hold-out (n = 1,122) | **0.8234** | 0.82 |

Random Forest and DistilBERT are effectively tied at the top, and all three models cluster tightly in the 0.81–0.82 macro-F1 range. *Stress* is the easiest class across all models (F1 ≈ 0.86–0.88); *Depression* is consistently the hardest (F1 ≈ 0.75–0.76) due to vocabulary overlap with *Anxiety* and *Personality disorder*.

## Dataset

**Source:** [Reddit Mental Health Dataset](https://www.kaggle.com/datasets/neelghoshal/reddit-mental-health-data) (Kaggle, Neel Ghoshal)

- 5,957 Reddit posts → 5,607 after dropping rows with missing `text` or `target`
- Five near-balanced classes (1,077–1,202 posts each)
- Columns: `text`, `title`, `target` (0–4)

### Downloading the data

1. Create a Kaggle account if you don't have one.
2. Install the Kaggle CLI: `pip install kaggle`
3. Place your `kaggle.json` API token in `~/.kaggle/`
4. Run:
   ```bash
   kaggle datasets download -d neelghoshal/reddit-mental-health-data
   unzip reddit-mental-health-data.zip
   ```
5. Place the resulting CSV in the project root, or update the file path in Section 2 of the notebook.

Alternatively, download the CSV manually from the Kaggle page above.

## Project Structure

```
.
├── AshaShah_CS590_Capstone.ipynb   
├── README.md                       
└── Images from code                
```

## Methodology

The notebook is organised into self-contained sections:

1. **Setup & Imports** : package installation and global helpers
2. **Data Loading & Cleaning** : null handling, title+body merge, regex normalisation
3. **Exploratory Data Analysis** : class distribution, post length, VADER sentiment, per-class word clouds
4. **Baseline Model** : TF-IDF (1–2 grams, 30k vocab) + Logistic Regression, 5-fold stratified CV
5. **Random Forest** : TF-IDF (10k vocab) + 300-tree RF, with Gini feature-importance analysis
6. **DistilBERT Fine-Tuning** *(CS590 extension)* : `distilbert-base-uncased`, AdamW, 3 epochs, max_length 256
7. **SHAP Explainability** *(CS590 extension)* : per-class feature attribution on the Logistic Regression model
8. **Results Summary & Conclusions**

## Reproducing the Results

```bash
# Clone the repo
git clone https://github.com/AshaShah/reddit-mental-health-classification.git
cd <reddit-mental-health-classification>

# Install dependencies
pip install pandas numpy matplotlib seaborn wordcloud scikit-learn \
            vaderSentiment shap transformers torch

# Download the dataset (see "Downloading the data" above)

# Open the notebook
jupyter notebook AshaShah_CS590_Capstone.ipynb
```


## Key Findings

- **Bag-of-words is highly competitive.** A 30k-feature TF-IDF + Logistic Regression baseline reaches macro F1 0.81; Random Forest and DistilBERT both reach 0.82. The transformer's contextual representations do not produce a meaningful improvement over a well-tuned TF-IDF pipeline for this single-domain, balanced dataset.
- **Feature importance and SHAP both confirm clinically plausible signals.** Top features per class are condition-name vocabulary (`stress`, `bipolar`, `avpd`, `anxiety`, `panic`) and clinical terms (`meds`, `episode`, `manic`, `diagnosed`) — not spurious artifacts.
- **Depression is the hardest class** in every model, consistently confused with *Anxiety* and *Personality disorder*. This is the most promising target for future fine-grained discrimination work.

## Limitations

The labels are subreddit-of-origin, not clinical diagnoses; macro F1 of 0.82 means roughly 1 in 5 predictions is wrong, which is too high for any unsupervised clinical use; Reddit demographics skew young, English-speaking, and online; and social-media language drifts over time.

## References

- Ghoshal, N. (2024). *Reddit Mental Health Dataset.* Kaggle.
- Sanh, V. et al. (2019). *DistilBERT, a distilled version of BERT.* arXiv:1910.01108.
- Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS 30.
- Hutto, C., & Gilbert, E. (2014). *VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text.* ICWSM.
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12, 2825–2830.

## License

Educational use only. Dataset usage is governed by the Kaggle dataset's own license.
