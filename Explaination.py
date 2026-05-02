import warnings  # استيراد مكتبة التحذيرات في بايثون
warnings.filterwarnings('ignore')  # تجاهل جميع التحذيرات التي قد تظهر أثناء تشغيل الكود للحفاظ على نظافة المخرجات

import pandas as pd  # استيراد مكتبة باندا للتعامل مع البيانات والجداول
import numpy as np  # استيراد مكتبة نامباي للعمليات الحسابية والمصفوفات
import matplotlib.pyplot as plt  # استيراد مكتبة ماتبلوتليب لرسم البيانات
import seaborn as sns  # استيراد مكتبة سيبورن لتحسين وتجميل الرسوم البيانية الإحصائية
import kagglehub  # استيراد مكتبة للوصول إلى بيانات منصة كاجل (Kaggle)
import pandas as pd  # استيراد مكتبة باندا مرة أخرى (سطر مكرر وغير ضروري ولكن لا يسبب خطأ)
import os  # استيراد مكتبة للتعامل مع مسارات الملفات ونظام التشغيل
from sklearn.preprocessing import LabelEncoder  # استيراد أداة لتحويل البيانات النصية (الفئات) إلى أرقام
from sklearn.preprocessing import LabelEncoder  # استيراد نفس الأداة مرة أخرى (سطر مكرر)
from sklearn.model_selection import train_test_split  # استيراد دالة لتقسيم البيانات إلى مجموعات تدريب واختبار
from sklearn.linear_model import LogisticRegression  # استيراد نموذج الانحدار اللوجستي للتصنيف
from sklearn.metrics import accuracy_score  # استيراد دالة لحساب دقة النموذج
from sklearn.model_selection import train_test_split  # استيراد دالة التقسيم مرة أخرى (سطر مكرر)
from sklearn.ensemble import RandomForestClassifier  # استيراد نموذج الغابة العشوائية للتصنيف
from sklearn.decomposition import PCA  # استيراد خوارزمية PCA لتقليل أبعاد البيانات
from sklearn.feature_selection import RFE  # استيراد خوارزمية RFE لاختيار أهم الميزات (المتغيرات)

path = kagglehub.dataset_download("georgesaavedra/news-popularity-in-social-media-platforms")  # تحميل مجموعة البيانات من كاجل وتخزين مسارها في متغير 'path'

file_path = os.path.join(path, "News_Final.csv")  # إنشاء المسار الكامل للملف عن طريق دمج مسار المجلد مع اسم الملف "News_Final.csv"


df = pd.read_csv(file_path)  # قراءة ملف CSV وتحميله إلى إطار بيانات (DataFrame) اسمه 'df'

df = df.dropna()  # حذف أي صفوف تحتوي على قيم مفقودة (NaN)
df = df.drop_duplicates()  # حذف الصفوف المكررة لضمان عدم تكرار البيانات


df['PublishDate'] = pd.to_datetime(df['PublishDate'])  # تحويل عمود تاريخ النشر إلى صيغة تاريخ ووقت يمكن التعامل معها
df['Year'] = df['PublishDate'].dt.year  # استخراج السنة من تاريخ النشر وإنشاء عمود جديد لها
df['Month'] = df['PublishDate'].dt.month  # استخراج الشهر من تاريخ النشر وإنشاء عمود جديد له
df['Hour'] = df['PublishDate'].dt.hour  # استخراج الساعة من تاريخ النشر وإنشاء عمود جديد لها


df['Popularity'] = (df['Facebook'] > 1000).astype(int)  # إنشاء عمود 'الشعبية': إذا كانت مشاركات الفيسبوك أكبر من 1000 تكون القيمة 1 (شائع)، وإلا 0 (غير شائع)

X = df[['SentimentTitle', 'SentimentHeadline', 'Topic', 'Source', 'GooglePlus', 'LinkedIn', 'Month', 'Hour']].copy()  # تحديد الأعمدة التي ستستخدم كمدخلات (ميزات) للنموذج وتخزينها في X
y = df['Popularity']  # تحديد عمود الهدف (الشعبية) وتخزينه في y


le = LabelEncoder()  # إنشاء كائن من أداة ترميز التسميات (LabelEncoder)
for col in X.select_dtypes(include=['object']).columns:  # المرور على كل عمود في X يحتوي على بيانات نصية (object)
    X[col] = le.fit_transform(X[col])  # تحويل القيم النصية في العمود الحالي إلى أرقام صحيحة


model_rfe = LogisticRegression(max_iter=1000)  # إنشاء نموذج انحدار لوجستي مؤقت لاستخدامه كأداة مساعدة في اختيار الميزات، مع زيادة عدد التكرارات المسموحة
rfe = RFE(model_rfe, n_features_to_select=3)  # إعداد خوارزمية RFE لاختيار أفضل 3 ميزات فقط بناءً على النموذج السابق
X_rfe = rfe.fit_transform(X, y)  # تطبيق عملية اختيار الميزات على البيانات والحصول على البيانات الجديدة المختصرة

selected_features_rfe = X.columns[rfe.support_]  # استخراج أسماء الأعمدة (الميزات) التي تم اختيارها بواسطة RFE
print("Selected Features using RFE:\n", selected_features_rfe)  # طباعة أسماء الميزات المختارة


X_train_base, X_test_base, y_train, y_test = train_test_split(X_rfe, y, test_size=0.2, random_state=42)  # تقسيم البيانات إلى تدريب (80%) واختبار (20%) بشكل عشوائي ثابت

rf_full = RandomForestClassifier(n_estimators=50, random_state=42)  # إنشاء نموذج غابة عشوائية يتكون من 50 شجرة قرار
rf_full.fit(X_train_base, y_train)  # تدريب النموذج على بيانات التدريب
acc_before = accuracy_score(y_test, rf_full.predict(X_test_base))  # حساب دقة النموذج على بيانات الاختبار قبل تطبيق تقليل الأبعاد الإضافي

print(f"Results BEFORE Dimension Reduction (Using {X_train_base.shape[1]} RFE-selected features):")  # طباعة عنوان يوضح أن هذه النتائج قبل تقليل الأبعاد باستخدام PCA
print(f"- Accuracy: {acc_before:.4f}")  # طباعة قيمة الدقة بأربعة منازل عشرية


pca = PCA(n_components=3)  # إعداد خوارزمية PCA لتقليل الأبعاد إلى 3 مكونات رئيسية
X_train_pca = pca.fit_transform(X_train_base)  # تطبيق PCA على بيانات التدريب وتحويلها
X_test_pca = pca.transform(X_test_base)  # تطبيق نفس تحويل PCA على بيانات الاختبار

rf_pca = RandomForestClassifier(n_estimators=50, random_state=42)  # إنشاء نموذج غابة عشوائية جديد بنفس الإعدادات
rf_pca.fit(X_train_pca, y_train)  # تدريب النموذج الجديد على البيانات بعد تقليل أبعادها بـ PCA
acc_after = accuracy_score(y_test, rf_pca.predict(X_test_pca))  # حساب دقة النموذج بعد تطبيق PCA

print(f"Results AFTER Dimension Reduction ({X_train_pca.shape[1]} PCA Components):")  # طباعة عنوان يوضح النتائج بعد تقليل الأبعاد
print(f"- Accuracy: {acc_after:.4f}")  # طباعة قيمة الدقة الجديدة



log_model = LogisticRegression(max_iter=1000)  # إنشاء نموذج انحدار لوجستي نهائي للتدريب
log_model.fit(X_train_base, y_train)  # تدريب نموذج الانحدار اللوجستي على بيانات التدريب (المختارة بـ RFE)


y_pred_log = log_model.predict(X_test_base)  # استخدام النموذج للتنبؤ بنتائج بيانات الاختبار
log_accuracy = accuracy_score(y_test, y_pred_log)  # حساب دقة نموذج الانحدار اللوجستي
print("Logistic Regression Accuracy: ", log_accuracy)  # طباعة دقة الانحدار اللوجستي


rf_model = RandomForestClassifier()  # إنشاء نموذج غابة عشوائية آخر (بالإعدادات الافتراضية)
rf_model.fit(X_train_base, y_train)  # تدريب هذا النموذج على بيانات التدريب


y_pred_rf = rf_model.predict(X_test_base)  # استخدام النموذج للتنبؤ بنتائج بيانات الاختبار
rf_accuracy = accuracy_score(y_test, y_pred_rf)  # حساب دقة نموذج الغابة العشوائية
print("Random Forest Accuracy: ", rf_accuracy)  # طباعة دقة الغابة العشوائية


df_numerical = df.select_dtypes(include=[np.number])  # استخراج الأعمدة الرقمية فقط من البيانات الأصلية (هذا المتغير لم يُستخدم لاحقاً في الكود المعروض)


plt.figure(figsize=(10, 6))  # إنشاء شكل بياني جديد بحجم 10 عرض و 6 ارتفاع
sns.countplot(x='Topic', hue='Popularity', data=df, palette='Set2')  # رسم مخطط عددي يوضح توزيع المواضيع حسب شعبيتها (ملون حسب الشعبية)
plt.title('Distribution of News by Topic and Popularity')  # وضع عنوان للرسم البياني
plt.xlabel('Topic')  # وضع تسمية للمحور السيني (المواضيع)
plt.ylabel('Count')  # وضع تسمية للمحور الصادي (العدد)
plt.xticks(rotation=45)  # تدوير نصوص المحور السيني بزاوية 45 درجة لتسهيل القراءة
plt.show()  # عرض الرسم البياني

labels = ['Logistic Regression', 'Random Forest']  # تعريف تسميات للنموذجين للمقارنة
accuracy_scores = [log_accuracy, rf_accuracy]  # قائمة تحتوي على قيم الدقة لكل نموذج

plt.figure(figsize=(7, 7))  # إنشاء شكل بياني جديد بحجم 7x7
plt.pie(accuracy_scores, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#99ff99'])  # رسم مخطط دائري (Pie Chart) يقارن بين دقتي النموذجين
plt.title('Model Accuracy Comparison')  # وضع عنوان للمخطط الدائري
plt.axis('equal')  # جعل المخطط الدائري دائرياً تماماً وليس بيضاوياً
plt.show()  # عرض المخطط الدائري

depths = [2, 5, 10, 15, 20]  # قائمة بقيم مختلفة لعمق الشجرة (Max Depth) لتجربتها
rf_accs = []  # قائمة فارغة لتخزين نتائج الدقة لكل عمق
for d in depths:  # حلقة تكرارية للمرور على كل قيمة عمق
    m = RandomForestClassifier(max_depth=d, n_estimators=50, random_state=42).fit(X_train_base, y_train)  # إنشاء وتدريب نموذج غابة عشوائية بالعمق الحالي
    rf_accs.append(accuracy_score(y_test, m.predict(X_test_base)))  # حساب الدقة وإضافتها للقائمة

plt.figure(figsize=(8, 4))  # إنشاء شكل بياني جديد بحجم 8 عرض و 4 ارتفاع
plt.plot(depths, rf_accs, marker='s', color='green', linewidth=2)  # رسم خط بياني يوضح العلاقة بين عمق الشجرة والدقة
plt.title('Hyperparameter Tuning: RF Accuracy vs Max Depth')  # وضع عنوان للرسم يوضح أنه ضبط للمعاملات
plt.xlabel('Max Depth Value')  # تسمية المحور السيني بقيمة العمق الأقصى
plt.ylabel('Test Accuracy')  # تسمية المحور الصادي بدقة الاختبار
plt.grid(True)  # إظهار خطوط الشبكة الخلفية لتسهيل القراءة
plt.show()  # عرض رسم بياني ضبط المعاملات