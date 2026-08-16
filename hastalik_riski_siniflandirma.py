"""
Meme Kanseri Riskinin Lojistik Regresyon ve KNN ile Sınıflandırılması
----------------------------------------------------------------------
Veri seti: Breast Cancer Wisconsin (Diagnostic) - scikit-learn built-in
Amaç: Hücre görüntülerinden çıkarılan özelliklere bakarak bir tümörün
      malign (kötü huylu) mu, benign (iyi huylu) mu olduğunu tahmin etmek.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score
)

sns.set_style("whitegrid")

# =====================================================================
# 1. VERİ SETİNİ YÜKLEME VE İLK İNCELEME
# =====================================================================
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target  # 0 = malign, 1 = benign

print("Veri seti boyutu:", df.shape)
print("Eksik değer sayısı:", df.isnull().sum().sum())
print("Hedef değişken dağılımı:\n", df["target"].value_counts())

# =====================================================================
# 2. VERİ ANALİZİ VE GÖRSELLEŞTİRME
# =====================================================================

# Hedef değişken dağılımı
plt.figure(figsize=(5, 4))
counts = df["target"].value_counts().sort_index()
plt.bar(["Malign", "Benign"], counts.values, color=["#e74c3c", "#2ecc71"])
plt.title("Hedef Değişken Dağılımı")
plt.ylabel("Örnek Sayısı")
plt.tight_layout()
plt.savefig("plot1_target_dist.png", dpi=130)
plt.close()

# Seçili değişkenlerin sınıflara göre histogramı
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
cols_to_plot = ["mean radius", "mean texture", "mean smoothness", "mean concavity"]
for ax, col in zip(axes.flatten(), cols_to_plot):
    ax.hist(df[df.target == 0][col], bins=25, alpha=0.6, label="Malign", color="#e74c3c")
    ax.hist(df[df.target == 1][col], bins=25, alpha=0.6, label="Benign", color="#2ecc71")
    ax.set_title(col)
    ax.legend()
plt.tight_layout()
plt.savefig("plot2_histograms.png", dpi=130)
plt.close()

# Korelasyon matrisi (ilk 10 özellik)
plt.figure(figsize=(9, 7))
sns.heatmap(df.iloc[:, :10].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("İlk 10 Özellik Arasındaki Korelasyon")
plt.tight_layout()
plt.savefig("plot3_correlation.png", dpi=130)
plt.close()

# Aykırı değer kontrolü
plt.figure(figsize=(10, 5))
df[["mean radius", "mean area", "mean perimeter", "mean concavity"]].boxplot()
plt.title("Seçili Değişkenlerde Aykırı Değer Kontrolü")
plt.tight_layout()
plt.savefig("plot4_boxplot.png", dpi=130)
plt.close()

# =====================================================================
# 3. VERİ ÖN İŞLEME
# =====================================================================
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================================
# 4. MODEL KURMA: LOJİSTİK REGRESYON VE KNN
# =====================================================================
results = {}

log_model = LogisticRegression(max_iter=5000, random_state=42)
log_model.fit(X_train_scaled, y_train)
y_pred_log = log_model.predict(X_test_scaled)
y_proba_log = log_model.predict_proba(X_test_scaled)[:, 1]

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)
y_proba_knn = knn_model.predict_proba(X_test_scaled)[:, 1]

for name, y_pred, y_proba in [
    ("Lojistik Regresyon", y_pred_log, y_proba_log),
    ("KNN (k=5)", y_pred_knn, y_proba_knn),
]:
    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_proba),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }
    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred, target_names=["Malign", "Benign"]))

# =====================================================================
# 5. SONUÇLARIN GÖRSELLEŞTİRİLMESİ
# =====================================================================

# Confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Malign", "Benign"], yticklabels=["Malign", "Benign"])
    ax.set_title(f"{name}\nConfusion Matrix")
plt.tight_layout()
plt.savefig("plot5_confusion_matrix.png", dpi=130)
plt.close()

# ROC eğrisi
plt.figure(figsize=(6, 5))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {res['auc']:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Eğrisi Karşılaştırması")
plt.legend()
plt.tight_layout()
plt.savefig("plot6_roc.png", dpi=130)
plt.close()

# En etkili değişkenler
coef_df = pd.DataFrame({
    "feature": X.columns, "coefficient": log_model.coef_[0]
}).sort_values("coefficient", key=abs, ascending=False).head(10)

plt.figure(figsize=(8, 5))
colors = ["#e74c3c" if c < 0 else "#2ecc71" for c in coef_df["coefficient"]]
plt.barh(coef_df["feature"], coef_df["coefficient"], color=colors)
plt.title("En Etkili 10 Değişken (Lojistik Regresyon Katsayıları)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("plot7_feature_importance.png", dpi=130)
plt.close()

# Model karşılaştırma tablosu
comp_df = pd.DataFrame({
    name: {k: v for k, v in res.items() if k not in ["y_pred", "y_proba"]}
    for name, res in results.items()
}).T
print("\n=== Model Karşılaştırma Tablosu ===")
print(comp_df.round(4))
