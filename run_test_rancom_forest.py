from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 加载数据集
iris = load_iris()
X = iris.data  # 特征
y = iris.target  # 标签

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 3. 创建随机森林分类器
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# 4. 拟合模型
rf_classifier.fit(X_train, y_train)

# 5. 进行预测
y_pred = rf_classifier.predict(X_test)

# 6. 输出分类报告
print("classification_report: ")
print(classification_report(y_test, y_pred))

# 7. 混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
print("confusion_matrix：")
print(conf_matrix)

# 8. 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.ylabel('GT')
plt.xlabel('Predicted')
plt.title('matrix')
plt.show()

# 9. 特征重要性
feature_importances = rf_classifier.feature_importances_
features = iris.feature_names

# 10. 可视化特征重要性
plt.figure(figsize=(8, 6))
sns.barplot(x=feature_importances, y=features)
plt.title('feature importance')
plt.show()
