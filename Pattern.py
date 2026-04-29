import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA 
from sklearn.feature_selection import RFE  

path = kagglehub.dataset_download("georgesaavedra/news-popularity-in-social-media-platforms")

file_path = os.path.join(path, "News_Final.csv") 


df = pd.read_csv(file_path)

df = df.dropna()  
df = df.drop_duplicates() 


df['PublishDate'] = pd.to_datetime(df['PublishDate'])
df['Year'] = df['PublishDate'].dt.year
df['Month'] = df['PublishDate'].dt.month
df['Hour'] = df['PublishDate'].dt.hour


df['Popularity'] = (df['Facebook'] > 1000).astype(int)  

X = df[['SentimentTitle', 'SentimentHeadline', 'Topic', 'Source', 'GooglePlus', 'LinkedIn', 'Month', 'Hour']].copy()
y = df['Popularity']


le = LabelEncoder()
for col in X.select_dtypes(include=['object']).columns:
    X[col] = le.fit_transform(X[col])


model_rfe = LogisticRegression(max_iter=1000) 
rfe = RFE(model_rfe, n_features_to_select=3)  
X_rfe = rfe.fit_transform(X, y)

selected_features_rfe = X.columns[rfe.support_]
print("Selected Features using RFE:\n", selected_features_rfe)


pca = PCA(n_components=3)  
X_pca = pca.fit_transform(X_rfe)  

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2) 



rf_full = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_train, y_train)
acc_before = accuracy_score(y_test, rf_full.predict(X_test))

print(f"Results BEFORE Dimension Reduction (8 Features):")
print(f"- Accuracy: {acc_before:.4f}")


pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

rf_pca = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_train_pca, y_train)
acc_after = accuracy_score(y_test, rf_pca.predict(X_test_pca))

print(f"Results AFTER Dimension Reduction (3 PCA Components):")
print(f"- Accuracy: {acc_after:.4f}")



log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)


y_pred_log = log_model.predict(X_test)
log_accuracy = accuracy_score(y_test, y_pred_log)
print("Logistic Regression Accuracy: ", log_accuracy)


rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)


y_pred_rf = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy: ", rf_accuracy)


df_numerical = df.select_dtypes(include=[np.number])  


plt.figure(figsize=(10, 6))
sns.countplot(x='Topic', hue='Popularity', data=df, palette='Set2')
plt.title('Distribution of News by Topic and Popularity')
plt.xlabel('Topic')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

labels = ['Logistic Regression', 'Random Forest']
accuracy_scores = [log_accuracy, rf_accuracy]

plt.figure(figsize=(7, 7))
plt.pie(accuracy_scores, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#99ff99'])
plt.title('Model Accuracy Comparison')
plt.axis('equal') 
plt.show()

depths = [2, 5, 10, 15, 20]
rf_accs = []
for d in depths:
    m = RandomForestClassifier(max_depth=d, n_estimators=50, random_state=42).fit(X_train, y_train)
    rf_accs.append(accuracy_score(y_test, m.predict(X_test)))

plt.figure(figsize=(8, 4))
plt.plot(depths, rf_accs, marker='s', color='green', linewidth=2)
plt.title('Hyperparameter Tuning: RF Accuracy vs Max Depth')
plt.xlabel('Max Depth Value')
plt.ylabel('Test Accuracy')
plt.grid(True)
plt.show()