# ml-pca-projects

Python ML projects and PCA implementations — dimensionality reduction, scikit-learn pipelines, NumPy from-scratch builds, and notebook experiments.

## Structure

\\\
ml-pca-projects/
├── src/
│   ├── pca/            # PCA implementations (from-scratch + sklearn)
│   ├── preprocessing/  # Scaling, encoding, cleaning
│   └── models/         # Pipeline factories and model wrappers
├── notebooks/          # Jupyter exploration and demos
├── data/
│   ├── raw/
│   └── processed/
├── tests/              # Pytest unit tests
├── .gitignore
├── requirements.txt
└── README.md
\\\

## Projects

| Project | Description | Status |
|---------|-------------|--------|
| PCA from Scratch | NumPy-only PCA via eigen decomposition | 🔨 In Progress |
| Sklearn PCA Pipeline | sklearn pipeline: scaler → PCA → classifier | 🔨 In Progress |

## Setup

\\\ash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
\\\

## Tech Stack
- Python 3.11+, NumPy, Pandas, SciPy
- Scikit-learn, XGBoost, LightGBM
- PyTorch, MLflow, SHAP
- Matplotlib, Seaborn, Plotly, Jupyter
