# ColumnTransformer — Revision Notes

## Why do we need ColumnTransformer?

Real-world datasets contain different types of columns:

| Column | Type | Required Preprocessing |
|----------|----------|----------|
| Age | Numerical | Scaling |
| Gender | Categorical (Nominal) | One-Hot Encoding |
| Education | Categorical (Ordinal) | Ordinal Encoding |
| Fever | Missing Values | Imputation |

Since each column needs different preprocessing, we use **ColumnTransformer**.

---

## Definition

`ColumnTransformer` allows us to apply different transformations to different columns of a dataset simultaneously.

```python
from sklearn.compose import ColumnTransformer
```

---

## General Syntax

```python
ColumnTransformer(
    transformers=[
        (name1, transformer1, columns1),
        (name2, transformer2, columns2),
        ...
    ],
    remainder='drop'
)
```

### Parameters

#### 1. name

Just a label.

```python
'tnf1'
```

Can be anything.

```python
'imputer'
'encoder'
'scaler'
```

---

#### 2. transformer

The preprocessing operation.

Examples:

```python
SimpleImputer()
OrdinalEncoder()
OneHotEncoder()
StandardScaler()
MinMaxScaler()
```

---

#### 3. columns

Column(s) on which the transformer should be applied.

Single column:

```python
['fever']
```

Multiple columns:

```python
['gender', 'city']
```

---

## Example

Dataset:

| fever | cough | gender | city | age |
|--------|--------|--------|--------|--------|
| High | Mild | Male | Delhi | 25 |
| NaN | Strong | Female | Mumbai | 30 |

### ColumnTransformer

```python
transformer = ColumnTransformer(
    transformers=[
        ('tnf1',
         SimpleImputer(strategy='most_frequent'),
         ['fever']),

        ('tnf2',
         OrdinalEncoder(categories=[['Mild','Strong']]),
         ['cough']),

        ('tnf3',
         OneHotEncoder(drop='first'),
         ['gender','city'])
    ],
    remainder='passthrough'
)
```

---

# Understanding Each Tuple

### First Tuple

```python
('tnf1',
 SimpleImputer(),
 ['fever'])
```

Apply imputation only on `fever`.

---

### Second Tuple

```python
('tnf2',
 OrdinalEncoder(categories=[['Mild','Strong']]),
 ['cough'])
```

Apply ordinal encoding on `cough`.

Result:

```text
Mild   → 0
Strong → 1
```

---

### Third Tuple

```python
('tnf3',
 OneHotEncoder(drop='first'),
 ['gender','city'])
```

Apply one-hot encoding on `gender` and `city`.

---

## Why is `categories` a 2D List?

For one column:

```python
OrdinalEncoder(
    categories=[
        ['Mild','Strong']
    ]
)
```

For two columns:

```python
OrdinalEncoder(
    categories=[
        ['Mild','Strong'],
        ['Low','Medium','High']
    ]
)
```

Pattern:

```python
[
    categories_for_column1,
    categories_for_column2,
    categories_for_column3
]
```

### Mapping Rule

```python
Columns:
['cough', 'fever']

Categories:
[
    ['Mild', 'Strong'],
    ['Low', 'Medium', 'High']
]
```

- First column ↔ First category list
- Second column ↔ Second category list

---

## `remainder` Parameter

### 1. Drop Remaining Columns (Default)

```python
remainder='drop'
```

Columns not mentioned are removed.

---

### 2. Keep Remaining Columns

```python
remainder='passthrough'
```

Columns not mentioned remain unchanged.

Example:

```python
age
```

passes through untouched.

---

## `fit_transform()` vs `transform()`

### Training Data

```python
X_train = transformer.fit_transform(X_train)
```

### What does `fit()` do?

Learns:

- Missing value strategy
- Category mappings
- Scaling statistics

### What does `transform()` do?

Applies those learned rules.

---

### Test Data

```python
X_test = transformer.transform(X_test)
```

Only transformation is applied.

❌ Never do:

```python
transformer.fit_transform(X_test)
```

because it causes **data leakage**.

---

## Shape Changes

Original:

| fever | cough | gender | city | age |
|--------|--------|--------|--------|--------|
| | | | | |

5 columns

After transformation:

- fever → 1 column
- cough → 1 column
- gender → 1 column (after `drop='first'`)
- city → multiple columns
- age → passthrough

Number of columns usually increases.

Example:

```python
X_train.shape
```

may become

```python
(80, 7)
```

instead of

```python
(80, 5)
```

---

## Order of Output Columns

Output columns follow the order of transformers.

```python
transformers=[
    tnf1,
    tnf2,
    tnf3
]
```

Output order:

```text
tnf1 output
tnf2 output
tnf3 output
passthrough columns
```

---

## Access Feature Names

After fitting:

```python
transformer.get_feature_names_out()
```

Example:

```python
[
 'tnf1__fever',
 'tnf2__cough',
 'tnf3__gender_Male',
 'tnf3__city_Mumbai',
 'remainder__age'
]
```

Useful when using `OneHotEncoder`.

---

## Common Transformers Used with ColumnTransformer

### Missing Values

```python
SimpleImputer()
```

### Ordinal Categories

```python
OrdinalEncoder()
```

### Nominal Categories

```python
OneHotEncoder()
```

### Numerical Scaling

```python
StandardScaler()
```

or

```python
MinMaxScaler()
```

---

## Complete Example

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

transformer = ColumnTransformer(
    transformers=[
        (
            'imputer',
            SimpleImputer(strategy='most_frequent'),
            ['fever']
        ),
        (
            'ordinal',
            OrdinalEncoder(categories=[['Mild', 'Strong']]),
            ['cough']
        ),
        (
            'onehot',
            OneHotEncoder(drop='first'),
            ['gender', 'city']
        )
    ],
    remainder='passthrough'
)

X_train = transformer.fit_transform(X_train)
X_test = transformer.transform(X_test)
```

---

# Quick Revision Sheet

| Concept | Purpose |
|----------|----------|
| ColumnTransformer | Apply different preprocessing to different columns |
| SimpleImputer | Fill missing values |
| OrdinalEncoder | Encode ordered categories |
| OneHotEncoder | Encode unordered categories |
| StandardScaler | Standardize numerical features |
| fit_transform() | Training data |
| transform() | Test data |
| remainder='drop' | Remove unspecified columns |
| remainder='passthrough' | Keep unspecified columns |
| categories=[...] | Define category order for OrdinalEncoder |
| get_feature_names_out() | Get transformed column names |

---

## Interview One-Liner

> **ColumnTransformer is used to apply different preprocessing techniques to different columns of a dataset in a single pipeline while ensuring consistent transformations during training and testing.**
