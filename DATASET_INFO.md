# Dataset

The dataset files are **not included** in this repository due to file size constraints.

## Download Instructions

1. Go to the Kaggle dataset page:  
   👉 [https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

2. Download the following two files:
   - `True.csv` — 21,417 real news articles (label = 1)
   - `Fake.csv` — 23,481 fake news articles (label = 0)

3. Place both files inside this `dataset/` folder:

```
dataset/
├── True.csv
├── Fake.csv
└── README.md   ← this file
```

## Dataset Summary

| File | Label | Samples |
|------|-------|---------|
| True.csv | Real News | 21,417 |
| Fake.csv | Fake News | 23,481 |
| **Total** | — | **44,898** |

**Columns:** `title`, `text`, `subject`, `date`
