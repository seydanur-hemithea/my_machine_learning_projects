# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 16:00:57 2025

@author: asus
"""

from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

digit = load_digits()
X = digit.data
y = digit.target


# Etiketleri binary hale getir
y_bin = label_binarize(digit.target, classes=range(10))
X_train, X_test, y_train_bin, y_test_bin = train_test_split(digit.data, y_bin, test_size=0.2, random_state=42)

# SVM modeli (probability=True gerekli değil burada)
classifier = OneVsRestClassifier(SVC(kernel="rbf", probability=True))
classifier.fit(X_train, y_train_bin)

# Tahmin olasılıkları
y_score = classifier.predict_proba(X_test)

# Precision-Recall hesaplama
precision = dict()
recall = dict()
average_precision = dict()

for i in range(10):
    precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
    average_precision[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])

# 🎨 Precision-Recall Curve çizimi
plt.figure(figsize=(10, 8))
colors = plt.cm.get_cmap("tab10", 10)

for i in range(10):
    plt.plot(recall[i], precision[i], lw=2, color=colors(i),
             label=f"Sınıf {i} (AP = {average_precision[i]:.2f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("SVM Precision-Recall Curve (Digits Dataset)")
plt.legend(loc="lower left")
plt.grid(True)
plt.tight_layout()
plt.savefig("svm_digits_precision_recall.png")  # 📁 Görseli kaydet
plt.show()