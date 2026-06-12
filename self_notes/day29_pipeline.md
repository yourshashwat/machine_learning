# Machine Learning Pipelines in Scikit-Learn

## What is a Pipeline?

A **Pipeline** is a mechanism in Scikit-Learn that chains multiple data processing and machine learning steps together.

The output of one step automatically becomes the input of the next step.

### General Flow

Input Data
↓
Missing Value Handling
↓
Encoding
↓
Scaling
↓
Feature Selection
↓
Model Training
↓
Prediction

Instead of writing each step separately, a Pipeline combines them into a single object.

---

# Why Do We Need Pipelines?

Without pipelines:

- Preprocessing steps must be written manually.
- Same code must be repeated for training and testing data.
- Same preprocessing must be repeated in production.
- Easy to make mistakes.
- Hard to maintain code.

With pipelines:

- Cleaner code.
- Reusable workflow.
- Easier deployment.
- Reduced risk of preprocessing mistakes.
- Supports cross-validation and hyperparameter tuning easily.

---

# Example Used in Video

Dataset: Titanic Dataset

Target Column:

- `Survived = 1` → Passenger survived
- `Survived = 0` → Passenger died

Goal:

Predict whether a passenger survives based on:

- Pclass
- Sex
- Age
- SibSp
- Parch
- Fare
- Embarked

---

# Part 1: Without Pipeline

## Step 1: Import Libraries

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
```

## Step 2: Load Dataset

```python
df = pd.read_csv('train.csv')
```

---

## Step 3: Drop Unnecessary Columns

```python
df.drop(columns=['PassengerId','Name','Ticket','Cabin'], inplace=True)
```

Remaining Columns:

- Survived
- Pclass
- Sex
- Age
- SibSp
- Parch
- Fare
- Embarked

---

## Step 4: Train-Test Split

```python
X = df.drop(columns=['Survived'])
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)
```

---

## Step 5: Handle Missing Values

Missing columns:

### Age

Use Mean Imputation

```python
si_age = SimpleImputer()
```

### Embarked

Use Most Frequent Value

```python
si_embarked = SimpleImputer(strategy='most_frequent')
```

Apply on:

```python
X_train_age = si_age.fit_transform(X_train[['Age']])
X_train_embarked = si_embarked.fit_transform(X_train[['Embarked']])
```

---

## Step 6: One Hot Encoding

Categorical Columns:

- Sex
- Embarked

```python
ohe_sex = OneHotEncoder(drop='first')
ohe_embarked = OneHotEncoder(drop='first')
```

Apply:

```python
X_train_sex = ohe_sex.fit_transform(...)
X_train_embarked = ohe_embarked.fit_transform(...)
```

---

## Step 7: Combine Everything

```python
np.concatenate(...)
```

All transformed features are manually joined.

This step is tedious and error-prone.

---

## Step 8: Train Model

```python
clf = DecisionTreeClassifier()

clf.fit(X_train_transformed, y_train)
```

---

## Step 9: Prediction

```python
y_pred = clf.predict(X_test_transformed)
```

---

## Step 10: Save Objects

Need to save:

1. Model
2. Age Imputer
3. Embarked Imputer
4. Sex Encoder
5. Embarked Encoder

```python
pickle.dump(...)
```

Problem:

Too many objects to manage.

---

# Problems Without Pipelines

## Repeated Code

Same preprocessing logic must be written again during deployment.

---

## High Chance of Errors

Must remember:

- Correct order of transformations
- Correct columns
- Correct encoder objects

Any mismatch causes wrong predictions.

---

## Difficult Maintenance

If preprocessing changes:

- Training code changes
- Production code changes

Both must stay synchronized.

---

# Part 2: Using Pipelines

Pipeline automates everything.

---

# Planned Workflow

Step 1:

Missing Value Imputation

↓

Step 2:

One Hot Encoding

↓

Step 3:

Scaling

↓

Step 4:

Feature Selection

↓

Step 5:

Decision Tree

---

# ColumnTransformer

Used when different columns require different preprocessing.

```python
from sklearn.compose import ColumnTransformer
```

---

# Transformer 1: Imputation

```python
trf1 = ColumnTransformer([
    ('impute_age',
     SimpleImputer(),
     [2]),

    ('impute_embarked',
     SimpleImputer(strategy='most_frequent'),
     [6])
], remainder='passthrough')
```

Purpose:

- Fill missing Age
- Fill missing Embarked

---

# Transformer 2: One Hot Encoding

```python
trf2 = ColumnTransformer([
    ('ohe',
     OneHotEncoder(
         sparse=False,
         handle_unknown='ignore'
     ),
     [1,6])
], remainder='passthrough')
```

Purpose:

Convert categorical values into numbers.

---

# Why handle_unknown='ignore'?

Suppose production receives:

```text
New category not seen during training
```

Without ignore:

Error occurs.

With ignore:

Encoder safely handles new category.

---

# Transformer 3: Scaling

```python
from sklearn.preprocessing import MinMaxScaler
```

```python
trf3 = ColumnTransformer([
    ('scale',
     MinMaxScaler(),
     slice(0,10))
])
```

Purpose:

Bring features to same range.

Range:

```text
0 → 1
```

---

# Transformer 4: Feature Selection

```python
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
```

```python
trf4 = SelectKBest(
    score_func=chi2,
    k=8
)
```

Purpose:

Keep only top 8 most important features.

Benefits:

- Removes noise
- Reduces dimensionality
- Improves efficiency

---

# Transformer 5: Model

```python
trf5 = DecisionTreeClassifier()
```

---

# Creating Pipeline

```python
from sklearn.pipeline import Pipeline
```

```python
pipe = Pipeline([
    ('trf1', trf1),
    ('trf2', trf2),
    ('trf3', trf3),
    ('trf4', trf4),
    ('trf5', trf5)
])
```

---

# Training Pipeline

```python
pipe.fit(X_train, y_train)
```

Pipeline automatically performs:

1. Imputation
2. Encoding
3. Scaling
4. Feature Selection
5. Model Training

No manual work required.

---

# Prediction

```python
y_pred = pipe.predict(X_test)
```

Pipeline automatically performs all preprocessing.

---

# fit() vs fit_transform()

## When Pipeline Ends With Model

Use:

```python
pipe.fit()
```

because model training happens.

---

## When Pipeline Contains Only Preprocessing

Use:

```python
pipe.fit_transform()
```

because only transformation is needed.

---

# Pipeline Visualization

```python
from sklearn import set_config

set_config(display='diagram')
```

Shows visual representation of pipeline.

Benefits:

- Easy debugging
- Easy understanding
- Better documentation

---

# Inspecting Pipeline

## View All Steps

```python
pipe.named_steps
```

Returns:

```python
{
 'trf1': ...,
 'trf2': ...,
 'trf3': ...,
 'trf4': ...,
 'trf5': ...
}
```

---

## Access Specific Step

```python
pipe.named_steps['trf1']
```

Useful for debugging.

---

# Cross Validation with Pipeline

```python
from sklearn.model_selection import cross_val_score
```

```python
cross_val_score(
    pipe,
    X_train,
    y_train,
    cv=5,
    scoring='accuracy'
)
```

Advantages:

- Preprocessing is applied correctly in every fold.
- Prevents data leakage.

---

# Hyperparameter Tuning

Using GridSearchCV

```python
from sklearn.model_selection import GridSearchCV
```

Example:

```python
params = {
    'trf5__max_depth':[1,2,3,4,5]
}
```

Notice:

```text
StepName__ParameterName
```

Double underscore is mandatory.

---

## Grid Search

```python
grid = GridSearchCV(
    pipe,
    params,
    cv=5
)
```

```python
grid.fit(X_train,y_train)
```

---

## Best Parameters

```python
grid.best_params_
```

---

## Best Score

```python
grid.best_score_
```

---

# Saving Pipeline

Without Pipeline:

Need to save many objects.

With Pipeline:

Only one object.

```python
pickle.dump(
    pipe,
    open('pipe.pkl','wb')
)
```

---

# Production Workflow

## Load Pipeline

```python
pipe = pickle.load(
    open('pipe.pkl','rb')
)
```

---

## Create New Passenger Input

```python
test_input = np.array([...])
```

---

## Prediction

```python
pipe.predict(test_input)
```

That's it.

No:

- Imputer loading
- Encoder loading
- Scaling code
- Manual transformations

Everything is inside the pipeline.

---

# Major Advantages of Pipelines

1. Cleaner code
2. Less repetition
3. Easier deployment
4. Lower chance of errors
5. Supports Cross Validation
6. Supports GridSearchCV
7. Easy debugging
8. Easy maintenance
9. Prevents preprocessing mismatch
10. Industry standard practice

---

# Interview Questions

### What is a Pipeline?

A Pipeline is a Scikit-Learn utility that chains preprocessing and model training steps into a single workflow.

---

### Why use Pipelines?

To automate preprocessing, reduce code duplication, and simplify deployment.

---

### Difference between Pipeline and make_pipeline?

`Pipeline`

```python
Pipeline([
    ('scaler', scaler),
    ('model', model)
])
```

Requires step names.

`make_pipeline`

```python
make_pipeline(
    scaler,
    model
)
```

Automatically generates names.

---

### Why use ColumnTransformer?

To apply different transformations to different columns.

---

### Why use handle_unknown='ignore'?

To prevent errors when unseen categories appear during prediction.

---

### Why are Pipelines important in Production?

Because the exact same preprocessing used during training is automatically applied during prediction, preventing inconsistencies and errors.

---

# One-Line Summary

A Pipeline combines preprocessing and model training into a single reusable object, making machine learning workflows cleaner, safer, and easier to deploy.
