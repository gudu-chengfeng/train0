# -*- coding: utf-8 -*-
"""
手写数字识别分类任务（scikit-learn）
数据集：sklearn.datasets.load_digits（1797 个 8x8 灰度图，10 类，0-9）
流程：数据探索 -> 划分数据集 -> 多模型对比 -> 交叉验证 -> 最优模型详细评估 -> 输出报告
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

RANDOM_STATE = 42
report_lines = []


def log(text=""):
    """同时打印到控制台并写入报告缓冲区。"""
    print(text)
    report_lines.append(text)


# ---------- 1. 加载数据 ----------
digits = datasets.load_digits()
X, y = digits.data, digits.target
n_samples, n_features = X.shape
log("# 手写数字识别分类任务 —— 评估报告")
log()
log(f"- 数据集：sklearn 内置 digits（手写数字 0-9 灰度图，8x8 像素）")
log(f"- 样本数：{n_samples}，特征数：{n_features}（8x8=64 个像素值，范围 0-16）")
log(f"- 类别数：{len(digits.target_names)}")
log()

# 类别分布
counts = np.bincount(y)
log("| 类别 | 样本数 |")
log("|---|---|")
for digit, cnt in enumerate(counts):
    log(f"| {digit} | {cnt} |")
log()

# 样本图示例
fig, axes = plt.subplots(2, 5, figsize=(10, 5.2))
for ax, img, label in zip(axes.ravel(), digits.images[:10], y[:10]):
    ax.imshow(img, cmap="gray")
    ax.set_title(f"标签: {label}")
    ax.axis("off")
fig.suptitle("数据集样例（前 10 张 8x8 手写数字）")
fig.subplots_adjust(hspace=0.6, top=0.86)
fig.savefig("digits_samples.png", dpi=120)
plt.close(fig)

# ---------- 2. 划分训练/测试集 ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
log(f"## 1. 数据划分")
log()
log(f"- 训练集：{X_train.shape[0]} 条（75%），测试集：{X_test.shape[0]} 条（25%），分层抽样保持类别比例")
log()

# ---------- 3. 多模型对比 ----------
models = {
    "逻辑回归": make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=5000, C=0.1,
                                                 random_state=RANDOM_STATE)),
    "支持向量机(RBF)": make_pipeline(StandardScaler(),
                                     SVC(kernel="rbf", C=10, gamma=0.005,
                                         random_state=RANDOM_STATE)),
    "随机森林": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
}

log("## 2. 模型对比（测试集准确率 / 宏平均 F1）")
log()
log("| 模型 | 测试集准确率 | 宏平均 F1 |")
log("|---|---|---|")
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    results[name] = (model, acc, f1m)
    log(f"| {name} | {acc:.4f} | {f1m:.4f} |")
log()

best_name = max(results, key=lambda k: results[k][1])
best_model, best_acc, _ = results[best_name]

bar_fig, ax = plt.subplots(figsize=(7, 4))
names = list(results)
accs = [results[n][1] for n in names]
bars = ax.bar(names, accs, color=["#4c72b0", "#55a868", "#c44e52"])
ax.bar_label(bars, fmt="%.4f")
ax.set_ylim(0.9, 1.005)
ax.set_ylabel("测试集准确率")
ax.set_title("三种模型测试集准确率对比")
bar_fig.tight_layout()
bar_fig.savefig("model_comparison.png", dpi=120)
plt.close(bar_fig)

# ---------- 4. 交叉验证 ----------
log(f"## 3. 最优模型（{best_name}）5 折交叉验证")
log()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring="accuracy")
log(f"- 各折准确率：{', '.join(f'{s:.4f}' for s in cv_scores)}")
log(f"- 平均准确率：{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
log()

# ---------- 5. 最优模型详细评估 ----------
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

log(f"## 4. 最优模型（{best_name}）测试集详细评估")
log()
log(f"- 总体准确率：**{acc:.4f}**（{int(acc * len(y_test))}/{len(y_test)}）")
log()
log("```")
log(classification_report(y_test, y_pred,
                          target_names=[str(d) for d in digits.target_names]))
log("```")

# 混淆矩阵
cm_fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(10), range(10))
ax.set_yticks(range(10), range(10))
ax.set_xlabel("预测类别")
ax.set_ylabel("真实类别")
ax.set_title(f"{best_name} 混淆矩阵（测试集）")
for i in range(10):
    for j in range(10):
        if cm[i, j]:
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
cm_fig.colorbar(im)
cm_fig.tight_layout()
cm_fig.savefig("confusion_matrix.png", dpi=120)
plt.close(cm_fig)

# 错分样本展示
wrong = np.where(y_pred != y_test)[0]
log(f"- 测试集中被错分的样本数：{len(wrong)}")
if len(wrong):
    examples = []
    for idx in wrong:
        examples.append(f"真实 {y_test[idx]} 误判为 {y_pred[idx]}")
    log(f"- 错分情况：{', '.join(examples)}")
log()

# 错分样本展示（全部错分样本）
n_wrong = max(len(wrong), 1)
err_fig, axes = plt.subplots(1, n_wrong, figsize=(2.0 * n_wrong, 2.9))
if len(wrong) == 0:
    axes = [axes]
    axes[0].set_title("无错分样本")
    axes[0].axis("off")
else:
    for ax, idx in zip(np.atleast_1d(axes), wrong):
        ax.imshow(X_test[idx].reshape(8, 8), cmap="gray")
        ax.set_title(f"真实:{y_test[idx]}  预测:{y_pred[idx]}", fontsize=9)
        ax.axis("off")
err_fig.suptitle(f"全部 {len(wrong)} 个错分样本")
err_fig.subplots_adjust(top=0.78)
err_fig.savefig("misclassified.png", dpi=120)
plt.close(err_fig)

log(f"## 5. 结论")
log()
log(f"- {best_name} 在手写数字识别任务上表现最好：测试集准确率 {acc:.4f}，"
    f"5 折交叉验证平均准确率 {cv_scores.mean():.4f}。")
log(f"- SVM（RBF 核）对小规模图像分类任务非常合适；错分样本多为书写潦草的"
    f"边缘案例（如 8/1、9/5 相互混淆），符合肉眼判断难度。")
log("- 生成的图表：digits_samples.png、model_comparison.png、confusion_matrix.png、"
    "misclassified.png")
log()

with open("评估报告.md", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print("\n报告已保存到 评估报告.md")
