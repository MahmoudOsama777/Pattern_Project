import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA  # إضافة PCA لتقليل الأبعاد



path = kagglehub.dataset_download("georgesaavedra/news-popularity-in-social-media-platforms")


file_path = os.path.join(path, "News_Final.csv") 


df = pd.read_csv(file_path)


df = df.dropna() 
df = df.drop_duplicates()  

df['Popularity'] = (df['Facebook'] > 1000).astype(int)


X = df[['SentimentTitle', 'SentimentHeadline', 'Topic', 'Source']] 
y = df['Popularity'] 


le = LabelEncoder()


for col in X.select_dtypes(include=['object']).columns:
    X[col] = le.fit_transform(X[col])


selector = SelectKBest(score_func=f_classif, k=3)
X_new = selector.fit_transform(X, y)

selected_features = X.columns[selector.get_support()]
print("Selected Features:\n", selected_features)


pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_new)


X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))


model2 = RandomForestClassifier()
model2.fit(X_train, y_train)

y_pred2 = model2.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred2))

df_numerical = df.select_dtypes(include=[np.number])

plt.figure(figsize=(10, 6))
sns.heatmap(df_numerical.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
