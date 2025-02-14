import pandas as pd
import numpy as np
np.random.seed(42)
ecg_normal = np.random.randint(60, 100, 500)  
ecg_abnormal = np.random.choice([40, 45, 130, 150, 160], 100)  
glucose_normal = np.random.randint(70, 140, 500)  
glucose_abnormal = np.random.choice([50, 55, 220, 250, 300], 100)
normal_data = pd.DataFrame({'ECG': ecg_normal, 'Glucose': glucose_normal, 'Label': 0})
abnormal_data = pd.DataFrame({'ECG': ecg_abnormal, 'Glucose': glucose_abnormal, 'Label': 1})
df = pd.concat([normal_data, abnormal_data]).sample(frac=1).reset_index(drop=True)
df.to_csv("health_data.csv", index=False)

print(df.head())  
