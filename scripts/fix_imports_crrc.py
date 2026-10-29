# -*- coding: utf-8 -*-
with open('scripts/evaluate_crrc_cohort.py', 'r', encoding='utf-8') as f:
    text = f.read().replace('\r\n', '\n')

old_block = """from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from temporal_dynamics.adapters import load_mendeley_temporal_dataset

# Ensure v3/src is on path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "v3" / "src"))

from temporal_dynamics.crrc_adapter import ("""

new_block = """from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

# Ensure v3/src and src are on path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "v3" / "src"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from temporal_dynamics.adapters import load_mendeley_temporal_dataset
from temporal_dynamics.crrc_adapter import ("""

assert old_block in text, "old_block not found"
text = text.replace(old_block, new_block)

with open('scripts/evaluate_crrc_cohort.py', 'w', encoding='utf-8', newline='\r\n') as f:
    f.write(text)
print("Fixed imports in scripts/evaluate_crrc_cohort.py")
