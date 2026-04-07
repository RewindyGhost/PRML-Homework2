import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 2. 3D月牙数据生成函数
def make_moons_3d(n_samples_per_class=500, noise=0.1):
    t = np.linspace(0, 2 * np.pi, n_samples_per_class)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)
    X = np.vstack([np.column_stack([x, y, z]), np.column_stack([-x, y - 1, -z])])
    y = np.hstack([np.zeros(n_samples_per_class), np.ones(n_samples_per_class)])
    X += np.random.normal(scale=noise, size=X.shape)
    return X, y

X_train, y_train = make_moons_3d(n_samples_per_class=500, noise=0.2)
X_test, y_test = make_moons_3d(n_samples_per_class=250, noise=0.2)

# ---------------------- 初始化5种分类模型 ----------------------
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "AdaBoost + DT": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3),
                                        n_estimators=50, random_state=42),
    "SVM(Linear)": SVC(kernel='linear', random_state=42),
    "SVM(RBF)": SVC(kernel='rbf', random_state=42),
    "SVM(Poly)": SVC(kernel='poly', degree=3, random_state=42)
}

# ---------------------- 训练模型 + 保存预测结果 ----------------------
y_preds = {}
accuracy_results = {}

print("=" * 60)
print("5种模型分类性能评估")
print("=" * 60)
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_preds[name] = y_pred
    acc = accuracy_score(y_test, y_pred)
    accuracy_results[name] = acc
    print(f"\n【{name}】")
    print(f"测试集准确率: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["C0", "C1"]))


# ====================== 可视化1：3D分类结果======================
def plot_3d_classification():
    fig = plt.figure(figsize=(20, 12))
    for idx, (name, y_pred) in enumerate(y_preds.items()):
        ax = fig.add_subplot(2, 3, idx + 1, projection='3d')


        correct_0 = (y_pred == y_test) & (y_test == 0)
        correct_1 = (y_pred == y_test) & (y_test == 1)
        incorrect = (y_pred != y_test)

        ax.scatter(X_test[correct_0, 0], X_test[correct_0, 1], X_test[correct_0, 2],
                   c='blue', label='正确分类 C0', s=20, alpha=0.7)

        ax.scatter(X_test[correct_1, 0], X_test[correct_1, 1], X_test[correct_1, 2],
                   c='gold', label='正确分类 C1', s=20, alpha=0.7)

        ax.scatter(X_test[incorrect, 0], X_test[incorrect, 1], X_test[incorrect, 2],
                   c='red', marker='x', label='分类错误', s=40, linewidth=1.5)

        ax.set_title(f'{name} \n3D测试集分类结果', fontsize=12)
        ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
        ax.legend()
    plt.tight_layout()
    plt.show()


# ====================== 可视化2：准确率对比柱状图 ======================
def plot_accuracy_compare():
    plt.figure(figsize=(10, 6))
    names = list(accuracy_results.keys())
    accs = list(accuracy_results.values())
    bars = plt.bar(names, accs, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{acc:.3f}', ha='center', fontsize=12)
    plt.ylim(0.4, 1.0)
    plt.ylabel('准确率')
    plt.title('5种模型测试集准确率对比')
    plt.xticks(rotation=10)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.show()


# ====================== 可视化3：混淆矩阵热力图 ======================
def plot_confusion_heatmap():
    fig = plt.figure(figsize=(18, 10))
    for idx, (name, y_pred) in enumerate(y_preds.items()):
        cm = confusion_matrix(y_test, y_pred)
        ax = fig.add_subplot(2, 3, idx + 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['C0', 'C1'], yticklabels=['C0', 'C1'])
        ax.set_title(f'{name} 混淆矩阵')
        ax.set_xlabel('预测标签'), ax.set_ylabel('真实标签')
    plt.tight_layout()
    plt.show()


# ====================== 可视化4：决策边界图 ======================
def plot_decision_boundary():
    fig = plt.figure(figsize=(20, 12))
    z_mean = np.mean(X_test[:, 2])
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    for idx, (name, model) in enumerate(models.items()):
        ax = fig.add_subplot(2, 3, idx + 1)
        Z = model.predict(np.c_[xx.ravel(), yy.ravel(), np.full_like(xx.ravel(), z_mean)])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', edgecolors='k')
        ax.set_title(f'{name} 决策边界（X-Y投影）')
        ax.set_xlabel('X'), ax.set_ylabel('Y')
    plt.tight_layout()
    plt.show()


# ---------------------- 生成所有图 ----------------------
if __name__ == '__main__':
    plot_3d_classification()
    plot_accuracy_compare()
    plot_confusion_heatmap()
    plot_decision_boundary()