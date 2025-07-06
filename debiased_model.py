import pandas as pd
import numpy as np
import tabulate
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
from sklearn.linear_model import LogisticRegression
from fairlearn.metrics import (
    demographic_parity_difference,
    equal_opportunity_difference,
    false_positive_rate_difference
)

df = pd.read_csv('plain_resumes.csv')

male_data = df[df["Gender"] == 1]  
female_data = df[df["Gender"] == 0]

male_train, male_test = train_test_split(
    male_data,
    test_size=0.5,
    random_state=42,
    stratify=male_data["HiringDecision"]
)

female_train, female_test = train_test_split(
    female_data,
    test_size=0.5,
    random_state=42,
    stratify=female_data["HiringDecision"]
)

X_train = pd.concat([male_train.drop("HiringDecision", axis=1), female_train.drop("HiringDecision", axis=1)])
y_train = pd.concat([male_train["HiringDecision"], female_train["HiringDecision"]])

X_test = pd.concat([male_test.drop("HiringDecision", axis=1), female_test.drop("HiringDecision", axis=1)])
y_test = pd.concat([male_test["HiringDecision"], female_test["HiringDecision"]])

test_genders = X_test['Gender']

text_clf = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression(class_weight='balanced', random_state=42))
])

text_clf.fit(X_train['ResumeText'], y_train)
text_preds = text_clf.predict(X_test['ResumeText'])
tfidf = text_clf.named_steps['tfidf']



X_train_tfidf = tfidf.transform(X_train['ResumeText']).toarray()
X_test_tfidf = tfidf.transform(X_test['ResumeText']).toarray()

mitigator = ExponentiatedGradient(
    LogisticRegression(max_iter=1000),
    constraints=EqualizedOdds(),
    eps=0.1
)
mitigator.fit(X_train_tfidf, y_train, sensitive_features=X_train['Gender'])


y_pred_debiased = mitigator.predict(X_test_tfidf)  



def calculate_metrics(y_true, y_pred, sensitive_features, model_name):
    """Calculate all evaluation metrics"""
    return {
        'Model': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'DP Difference': demographic_parity_difference(
            y_true=y_true, 
            y_pred=y_pred, 
            sensitive_features=sensitive_features  # Now properly passed as named argument
        ),
        'EO Difference': equal_opportunity_difference(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features
        ),
        'Avg Odds Difference': 0.5 * (
            equal_opportunity_difference(
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_features
            ) + 
            false_positive_rate_difference(
                y_true=y_true,
                y_pred=y_pred, 
                sensitive_features=sensitive_features
            )
        )
    }

results = pd.DataFrame([
    calculate_metrics(
        y_true=y_test,
        y_pred=text_preds, 
        sensitive_features=test_genders,
        model_name="Original Model"
    ),
    calculate_metrics(
        y_true=y_test,
        y_pred=y_pred_debiased,
        sensitive_features=test_genders,
        model_name="Debiased Model"
    )
])

print("\nFairness-Accuracy Comparison:")
print(results[['Model', 'Accuracy', 'DP Difference', 'EO Difference', 'Avg Odds Difference']].to_markdown(index=False))

plt.figure(figsize=(10, 5))
metrics_to_plot = ['DP Difference', 'EO Difference', 'Avg Odds Difference']

for i, metric in enumerate(metrics_to_plot):
    plt.subplot(1, 3, i+1)
    sns.barplot(x='Model', y=metric, data=results)
    plt.axhline(0, color='k', linestyle='--')
    plt.title(metric)
    plt.xticks(rotation=45)
    
    for j, val in enumerate(results[metric]):
        plt.text(j, val + 0.01 if val > 0 else val - 0.02, 
                f"{val:.3f}", ha='center')

plt.tight_layout()
plt.savefig('fairness_metrics_debiased.png', dpi=300, bbox_inches='tight')  

plt.show()