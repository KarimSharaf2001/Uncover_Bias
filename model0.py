import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from fairlearn.metrics import demographic_parity_difference,false_positive_rate_difference,equal_opportunity_difference
from lime.lime_text import LimeTextExplainer
import random

df = pd.read_csv('plain_resumes.csv')

male_data = df[df["Gender"] == 1]  
female_data = df[df["Gender"] == 0]

male_train, male_test = train_test_split(
    male_data,
    test_size=0.3,
    random_state=42,
    stratify=male_data["HiringDecision"]
)

female_train, female_test = train_test_split(
    female_data,
    test_size=0.7,
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

print("\n[Baseline Model - No Debiasing]")
print("Accuracy:", accuracy_score(y_test, text_preds))
print(classification_report(y_test, text_preds))


def calculate_average_odds_difference(y_true, y_pred, sensitive_features):
    """Calculate average odds difference manually"""
    tpr_diff = equal_opportunity_difference(y_true, y_pred, sensitive_features=sensitive_features)
    fpr_diff = false_positive_rate_difference(y_true, y_pred, sensitive_features=sensitive_features)
    return (tpr_diff + fpr_diff) / 2

fairness_metrics = {
    'Demographic Parity': demographic_parity_difference(
        y_test, text_preds, sensitive_features=test_genders
    ),
    'Equal Opportunity': equal_opportunity_difference(
        y_test, text_preds, sensitive_features=test_genders
    ),
    'Average Odds Difference': calculate_average_odds_difference(
        y_test, text_preds, sensitive_features=test_genders
    )
}

print("\nFairness Metrics:")
for metric, value in fairness_metrics.items():
    print(f"{metric:>22}: {value:.4f}")

print("\nInterpretation:")
print("Positive values indicate bias toward male applicants")
print("Negative values indicate bias toward female applicants")
print("Zero indicates perfect fairness")

plt.figure(figsize=(12, 5))

results_df = pd.DataFrame({
    'Gender': test_genders.map({0: 'Female', 1: 'Male'}),
    'Prediction': text_preds,
    'Actual': y_test
})

plt.subplot(1, 2, 1)
sns.barplot(
    x='Gender', 
    y='Prediction', 
    data=results_df,
    estimator=lambda x: sum(x)/len(x)
)
plt.title('Demographic Parity\n(Selection Rates by Gender)')
plt.ylabel('Hiring Rate')
plt.ylim(0, 1)

plt.subplot(1, 2, 2)
sns.barplot(
    x='Gender', 
    y='Prediction', 
    data=results_df[results_df['Actual'] == 1],  
    estimator=lambda x: sum(x)/len(x)
)
plt.title('Equal Opportunity\n(True Positive Rates by Gender)')
plt.ylabel('True Positive Rate')
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig('fairness_metrics0.png', dpi=300, bbox_inches='tight')  

plt.show()

plt.figure(figsize=(8, 4))
plt.bar(fairness_metrics.keys(), fairness_metrics.values())
plt.axhline(0, color='black', linestyle='--')
plt.title("Fairness Metrics Comparison")
plt.ylabel("Difference (Male - Female)")
for i, v in enumerate(fairness_metrics.values()):
    plt.text(i, v + 0.01 if v > 0 else v - 0.02, f"{v:.3f}", ha='center')
    plt.savefig('fairness_metrics1.png', dpi=300, bbox_inches='tight')  

plt.show()

explainer = LimeTextExplainer(
    class_names=['Not Hire', 'Hire'],
    kernel_width=25, 
    random_state=42
)

predict_fn = lambda x: text_clf.named_steps['clf'].predict_proba(
    text_clf.named_steps['tfidf'].transform(x)
)

hire_indices = np.where(text_preds == 1)[0]
no_hire_indices = np.where(text_preds == 0)[0]

selected_cases = (
    random.sample(list(hire_indices), 3) + 
    random.sample(list(no_hire_indices), 2)
)

gender_keywords = {'male', 'female', 'he', 'she'}
gender_influence = []

for i, case_idx in enumerate(selected_cases):
    text = X_test['ResumeText'].iloc[case_idx]
    true_label = y_test.iloc[case_idx]
    pred_label = text_preds[case_idx]
    gender = "Male" if test_genders.iloc[case_idx] == 1 else "Female"
    
    print(f"\n{'='*50}")
    print(f"Case {i+1}: {'Hire' if pred_label == 1 else 'No Hire'}")
    print(f"True Label: {'Hire' if true_label == 1 else 'No Hire'}")
    print(f"Gender: {gender}")
    print("\nResume Excerpt:\n", text[:500] + "...")
    
    exp = explainer.explain_instance(
        text,
        predict_fn,
        num_features=10,
        top_labels=1
    )
    
    print("\nTop Influential Features:")
    features = exp.as_list(label=pred_label)
    for feature, weight in features:
        print(f"{feature:50} {weight:.3f}")
    
    case_gender_influence = any(
        any(keyword in feature.lower() for keyword in gender_keywords)
        for feature, _ in features
    )
    gender_influence.append(case_gender_influence)
    
    fig = exp.as_pyplot_figure(label=pred_label)
    plt.title(f"Case {i+1} ({gender})", pad=20)
    plt.tight_layout()
    plt.savefig(f'lime_case_{i+1}.png', bbox_inches='tight')
    plt.close()

print("\nGender Keyword Influence Summary:")
print(f"{sum(gender_influence)}/{len(selected_cases)} cases showed gender-related terms in top features")
if any(gender_influence):
    print("Gender terms appear to influence decisions")
else:
    print("No direct gender term influence detected in top features")

