#第2个版本
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from Model import SimplifiedCNN
from sklearn.metrics import r2_score,mean_squared_error,adjusted_rand_score,silhouette_samples
from scipy.stats import chi2
from matplotlib.patches import Ellipse

def load_array(data_arrays, batch_size, is_train=True):
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return DataLoader(dataset, batch_size, shuffle=is_train)

def plot_ellipse(ax, mean, cov, color, alpha=0.2):
    # 计算协方差矩阵的特征值和特征向量
    eigvals, eigvecs = np.linalg.eigh(cov)
    # 绘制95%置信区间椭圆 (alpha = 0.95)
    chi2_val = chi2.ppf(0.95, 2)  # 2维数据的卡方分布值
    # 椭圆的宽度和高度
    width, height = 2 * np.sqrt(eigvals * chi2_val)
    # 椭圆的旋转角度
    angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
    # 创建椭圆
    ellipse = Ellipse(mean, width, height, angle=np.degrees(angle), color=color, alpha=alpha)
    ax.add_patch(ellipse)

def main():
    # 文件路径
    model_dir = r"D:\python_project\pythonProject1\TAM+SEARS\train_log\average_Mcell_modelCONV3_bt50_e50_1218_SimplifiedCNN-Temporary_12-18_20-10.pth"
    data_path = r"E:\Study\yjs\y1\data\average_Mcell_test.xlsx"
    batch_size = 12

    # 读取数据
    df = pd.read_excel(data_path)
    X_test_Nom = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
    Y_test = torch.tensor(df.iloc[:, -1].values, dtype=torch.long)
    print('读取后的光谱数据完毕')
    print('真实数据标签：',Y_test)
    # 创建DataLoader
    test_loader = load_array((X_test_Nom, Y_test), batch_size, is_train=False)

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimplifiedCNN().to(device)

    # 加载模型
    model.load_state_dict(torch.load(model_dir))
    model.eval()  # 切换到评估模式

    # 进行预测
    predictions = []
    probabilities = []
    all_outputs = []
    with torch.no_grad():
        for data in test_loader:
            inputs, _ = data  # 获取输入特征和标签（标签在这里不需要）
            output = model(inputs.unsqueeze(1))
            # 存储每一批次的原始输出（logits 或 probabilities）
            all_outputs.extend(output.cpu().numpy())

            predictions.extend(output.argmax(dim=1).cpu().numpy().astype(int))
            probabilities.extend(torch.softmax(output, dim=1).cpu().numpy())

    # 打印预测结果
    print("预测结果:", predictions)
    # print('预测概率：', probabilities)

    #R指标
    r2 = r2_score(Y_test,predictions)
    r2 = ('%.2f' % r2)
    print("R方指标为：", r2)
    mse = mean_squared_error(Y_test, predictions)  # 均方误差
    print("MSE  : ", round(mse, 4))
    rmse = np.sqrt(mse)
    print("RMSE : ", round(rmse, 4))  # 均方根误差

    # 计算混淆矩阵
    cm = confusion_matrix(Y_test.cpu().numpy(), predictions)
    print("混淆矩阵:\n", cm)
    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(Y_test.cpu().numpy()),
                yticklabels=np.unique(Y_test.cpu().numpy()))
    plt.xlabel('Prediction')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    # 打印分类报告
    print(classification_report(Y_test.cpu().numpy(), predictions))

    # t-SNE 降维可视化
    out = np.array(all_outputs)
    n_samples = out.shape[0]  # 获取样本数量
    perplexity_value = min(30, n_samples - 1)  # 设置 perplexity 小于样本数
    print("Perplexity:",perplexity_value)

    # 2D
    X_embedded = TSNE(n_components=2,perplexity=20).fit_transform(out)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=predictions, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization')
    plt.xlabel('t-SNE D1')
    plt.ylabel('t-SNE D2')
    plt.show()

    # 创建t-SNE图
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    # 按类别分组数据并绘制每个类别的椭圆
    for label in np.unique(predictions):
        class_data = X_embedded[predictions == label]
        # 计算均值和协方差矩阵
        mean = np.mean(class_data, axis=0)
        cov = np.cov(class_data, rowvar=False)
        # 绘制椭圆
        plot_ellipse(ax, mean, cov, color=plt.cm.viridis(label / 3), alpha=0.2)
    # 绘制散点图
    scatter = ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=predictions, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization with 95% Confidence Ellipses')
    plt.xlabel('t-SNE D1')
    plt.ylabel('t-SNE D2')
    plt.show()

    #聚类中心计算
    kmeans = KMeans(n_clusters=3, random_state=42)#进行3聚类
    kmeans.fit(X_embedded)
    # 获取每个数据点的聚类标签
    labels = kmeans.labels_
    print("每个数据点的聚类标签：", labels)
    cluster_centers = kmeans.cluster_centers_
    print("聚类中心位置：\n", cluster_centers)
    distances = np.linalg.norm(X_embedded[:, np.newaxis] - cluster_centers, axis=2)
    print("每个数据点到每个聚类中心的距离：\n", distances)
    for i, label in enumerate(labels):
        print(f"数据点 {i} 属于聚类 {label}，到聚类 0 的距离：{distances[i, 0]:.2f}, "
              f"到聚类 1 的距离：{distances[i, 1]:.2f}, 到聚类 2 的距离：{distances[i, 2]:.2f}")
    min_distances = np.min(distances, axis=1)
    print("每个数据点到最近聚类中心的最小距离：", min_distances)
    # 绘制数据点，按照标签着色
    plt.figure(figsize=(10, 8))
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='X', s=200, label='Cluster Centers')
    for i, center in enumerate(cluster_centers):
        plt.annotate(f'Cluster {i}', (center[0], center[1]), fontsize=12, color='red',
                     horizontalalignment='center', verticalalignment='center')
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=predictions, cmap='viridis', alpha=0.6)
    for i, label in enumerate(labels):
        distance = distances[i, label]
        plt.annotate(f'{distance:.2f}', (X_embedded[i, 0], X_embedded[i, 1]), fontsize=8, color='black')

    #聚类评估定性
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization with Clustering')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.show()
    silhouette_values = silhouette_samples(X_embedded, labels)
    for cluster_id in range(kmeans.n_clusters):
        cluster_silhouette_values = silhouette_values[labels == cluster_id]
        cluster_avg_silhouette = cluster_silhouette_values.mean()
        print(f"聚类 {cluster_id} 的平均轮廓系数: {cluster_avg_silhouette:.4f}")

    ari = adjusted_rand_score(Y_test, labels)
    print("调整后的兰德指数：", ari)

    # AUC 计算
    true_labels_bin = (Y_test.cpu().numpy() == 1).astype(int)  # 将标签转为二进制形式
    fpr, tpr, thresholds = roc_curve(true_labels_bin, [p[1] for p in probabilities])  # 计算ROC曲线
    roc_auc = auc(fpr, tpr)  # 计算AUC值
    print('AUC: ', roc_auc)

    # 绘制ROC曲线
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    main()
