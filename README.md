# Titanic-dataset-with-LightgGBM
This project uses the **Titanic dataset** to predict passenger survival. It leverages a clean machine learning pipeline with **LightGBM**, and fine-tunes the model using **GridSearchCV**. The model also outputs **feature importances** to interpret what matters most.

---

## 🔍 Overview

Using the classic Titanic dataset from [Kaggle](https://www.kaggle.com/competitions/titanic), we:

- Clean and preprocess the data
- Use **LightGBM** for classification
- Tune hyperparameters with **GridSearchCV**
- Evaluate using accuracy, F1 score, and AUC
- Visualize feature importance

---

## 📦 Tech Stack

- Python
- Pandas, NumPy
- LightGBM
- scikit-learn
- Matplotlib, Seaborn

---

## 🧠 Model Pipeline Overview

1. Drop irrelevant features (`PassengerId`, `Name`, `Ticket`, `Cabin`)
2. Handle missing values (e.g. Age, Embarked)
3. Encode categorical features (`Sex`, `Embarked`, etc.)
4. Scale numeric features (optional)
5. Train/test split
6. Train **LightGBM classifier**
7. Tune with `GridSearchCV`
8. Evaluate with multiple metrics
9. Plot feature importances

---

## 📊 Model Metrics

The model is evaluated using:
- ✅ Accuracy
- 🧪 F1 Score
- 📈 ROC-AUC Score

Example output (varies by tuning):
Best Parameters: {'num_leaves': 31, 'learning_rate': 0.05}
Accuracy: 0.82
F1 Score: 0.78
AUC Score: 0.87

yaml
Copy
Edit

---

## 📈 Feature Importance Example

Top features may include:
- `Sex`
- `Fare`
- `Pclass`
- `Age`

![Feature Importance Example](#) <!-- Add a screenshot of your plot if available -->

---

## 🚀 How to Run

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/titanic-lightgbm-pipeline.git
cd titanic-lightgbm-pipeline
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
If you’re using Jupyter:

bash
Copy
Edit
jupyter notebook
3. Run the Notebook
Open the notebook and run all cells in order.

🗂 Example Files
bash
Copy
Edit
titanic-lightgbm-pipeline/
├── titanic_lgb_pipeline.ipynb     # Main notebook
├── data/                          # (Optional) CSV dataset
├── README.md                      # This file
└── requirements.txt
📥 Dataset
Get the dataset from:
Kaggle Titanic Competition

Place train.csv in a data/ folder or update the path in your notebook.

Author --- Nwankwo Emmanuel Ota
