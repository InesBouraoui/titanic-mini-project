import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

np.random.seed(42)

# load dataset
df = pd.read_csv("train.csv")

# basic cleaning
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# feature engineering
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

# encode categorical
df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

features = [
    "Pclass","Age","Fare","SibSp","Parch",
    "FamilySize","IsAlone",
    "Sex_male","Embarked_Q","Embarked_S"
]
X = df[features]
y = df["Survived"]

# train / validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_val)

f1 = f1_score(y_val, preds)

print("Validation F1:", f1)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_val, preds)
print("Confusion matrix:")
print(cm)
