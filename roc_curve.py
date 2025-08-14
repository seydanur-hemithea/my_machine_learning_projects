# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 15:59:17 2025

@author: asus
"""

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import numpy as np

# Veri seti
digits = load_digits()
X = digits.data
y = digits.target
n_classes = len(np.unique(y))

# Etiketleri binary hale getir (One-vs-Rest için)
y_bin = label_binarize(y, classes=range(n_classes))

# Eğitim-test bölme
X_train, X_test, y_train_bin, y_test_bin = train_test_split(X, y_bin, test_size=0.2, random_state=42)

# SVM modeli (probability=True ROC için gerekli)
classifier = OneVsRestClassifier(SVC(kernel="rbf", probability=True, random_state=42))
classifier.fit(X_train, y_train_bin)

# Tahmin olasılıkları
y_score = classifier.predict_proba(X_test)

# ROC Curve hesaplama
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 🎨 ROC Curve çizimi
plt.figure(figsize=(10, 8))
colors = plt.cm.get_cmap("tab10", n_classes)

for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], color=colors(i), lw=2,
             label=f"Sınıf {i} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], "k--", lw=2, label="Rastgele Tahmin")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("SVM ROC Curve (Digits Dataset)")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("svm_digits_roc_curve.png")  # 📁 Görseli kaydet
plt.show()
