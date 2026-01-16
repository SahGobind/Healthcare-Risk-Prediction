import pandas as pd
import numpy as np

# Data Processing
patients = pd.read_csv("patients.csv")
diagnoses = pd.read_csv("diagnoses.csv")
labs = pd.read_csv("labs.csv")
outcomes = pd.read_csv("outcomes.csv")

patients['AdmissionDate'] = pd.to_datetime(patients['AdmissionDate'])
patients['DischargeDate'] = pd.to_datetime(patients['DischargeDate'])
patients['LengthOfStay'] = (patients['DischargeDate'] - patients['AdmissionDate']).dt.days

patients = patients.merge(diagnoses, on='DiagnosisID')
patients = patients.merge(outcomes, on='OutcomeID')

patients['OutcomeEncoded'] = patients['OutcomeName'].map({
    'Recovered': 0,
    'Complicated': 1,
    'Deceased': 1
})

patients['HighRisk'] = np.where(
    (patients['Age'] > 65) &
    (patients['OutcomeName'].isin(['Complicated', 'Deceased'])),
    1, 0
)

abnormal_conditions = {
    'Blood Sugar': lambda x: x > 120,
    'Cholesterol': lambda x: x > 200,
    'Hemoglobin': lambda x: x < 13
}

def count_abnormal_labs(patient_id):
    patient_labs = labs[labs['PatientID'] == patient_id]
    count = 0
    for test_name, condition in abnormal_conditions.items():
        test_results = patient_labs[patient_labs['TestName'] == test_name]
        count += test_results['Result'].apply(condition).sum()
    return count

patients['AbnormalLabCount'] = patients['PatientID'].apply(count_abnormal_labs)

# Model Training
features = patients[['Age', 'LengthOfStay', 'TreatmentCost', 'AbnormalLabCount']]
target = patients['OutcomeEncoded']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.3, random_state=42
)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix
y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

# ROC Curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristics')
plt.legend(loc='lower right')
plt.show()


