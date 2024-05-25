import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Veriyi oku
df = pd.read_csv('breast_cancer.csv')

# Sınıf etiketlerini ve özellikleri belirle
X = df.iloc[:, :-1]  # Özellikler (features)
y = df.iloc[:, -1]   # Sınıflar (benign veya malignant)

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli tanımla ve eğit
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Test seti ile tahminlerde bulun
y_pred = model.predict(X_test)

# Sınıf etiketlerini kontrol et
print(y.unique())  # [2 4] çıktı

# Performans metriklerini hesaplama
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, pos_label=4)  # Doğru sınıf etiketini kullanın
precision = precision_score(y_test, y_pred, pos_label=4)
recall = recall_score(y_test, y_pred, pos_label=4)

# Sonuçları yazdırma
print(f'Accuracy: {accuracy:.2f}')
print(f'F1 Score: {f1:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')

# Performans metriklerini tabloya dönüştürme
metrics = {
    'Metric': ['Accuracy', 'F1 Score', 'Precision', 'Recall'],
    'Value': [accuracy, f1, precision, recall]
}

# DataFrame oluşturma
metrics_df = pd.DataFrame(metrics)

# Performans metriklerinin çubuk grafiğini oluşturma
plt.figure(figsize=(8, 6))
sns.barplot(x='Metric', y='Value', data=metrics_df, palette='viridis')
plt.ylim(0, 1)
plt.title('Model Performans Metrikleri')
plt.ylabel('Değer')
plt.xlabel('Metri')
plt.show()

# Karışıklık Matrisi (Confusion Matrix)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.title('Karışıklık Matrisi')
plt.show()

# ROC Eğrisi ve AUC (Area Under Curve) Hesaplama
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label=4)
roc_auc = roc_auc_score(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC eğrisi (alan = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Yanlış Pozitif Oranı (False Positive Rate)')
plt.ylabel('Doğru Pozitif Oranı (True Positive Rate)')
plt.title('ROC Eğrisi')
plt.legend(loc="lower right")
plt.show()
