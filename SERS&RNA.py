import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ==================== 1. 数据加载与预处理 ====================
def load_and_preprocess_data(gene_path, raman_path):
    """加载并预处理基因表达和拉曼数据"""
    try:
        # 读取数据
        df_gene = pd.read_excel(gene_path, header=0, index_col=0)
        df_raman = pd.read_excel(raman_path, header=0, index_col=0)

        # 检查样本一致性
        if df_gene.shape[0] != df_raman.shape[0]:
            raise ValueError(f"样本数量不一致！基因数据: {df_gene.shape[0]}, 拉曼数据: {df_raman.shape[0]}")

        # 确保样本顺序一致
        if not all(df_gene.index == df_raman.index):
            print("警告：样本顺序不一致，将按索引自动对齐！")
            df_gene = df_gene.reindex(df_raman.index)

        # 确保列名是字符串类型
        df_raman.columns = df_raman.columns.astype(str)

        return df_raman, df_gene
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None, None


# ==================== 2. 相关性分析与筛选 ====================
def calculate_correlation(raman_data, gene_data, threshold=0.8):
    """计算每个拉曼位移（列）与每个基因（列）之间的 Pearson 相关系数"""

    correlation_results = []

    # 计算每个拉曼位移与每个基因的相关性
    for raman_col in raman_data.columns:
        for gene_col in gene_data.columns:
            r = raman_data[raman_col].corr(gene_data[gene_col])
            correlation_results.append((raman_col, gene_col, r))

    # 创建相关性数据框
    correlation_df = pd.DataFrame(correlation_results, columns=["Raman_Shift", "Gene", "Pearson_r"])

    # 生成相关性矩阵
    corr_matrix = correlation_df.pivot(index='Gene', columns='Raman_Shift', values='Pearson_r')

    # 筛选出显著相关对
    significant_pairs = correlation_df[correlation_df['Pearson_r'].abs() > threshold]

    return corr_matrix, significant_pairs


# ==================== 3. 优化可视化 ====================
def plot_correlation_heatmap(corr_matrix, significant_pairs, title="Raman-Gene Correlation Heatmap"):
    """绘制优化的热图，只显示显著相关的部分"""
    if significant_pairs.empty:
        print("没有显著相关的基因，无法绘制热图")
        return None

    # 创建图形
    fig, (ax_heatmap, ax_colorbar) = plt.subplots(
        1, 2, figsize=(20, 15),
        gridspec_kw={'width_ratios': [0.95, 0.05]}
    )

    # 提取显著相关的基因和拉曼位移
    significant_genes = significant_pairs['Gene'].unique()
    significant_raman = significant_pairs['Raman_Shift'].unique()

    # 创建只包含显著相关对的矩阵
    significant_corr_matrix = corr_matrix.loc[significant_genes, significant_raman]

    # 绘制热图
    sns.heatmap(significant_corr_matrix,  # 使用筛选后的相关性矩阵
                cmap='coolwarm',
                center=0,
                vmin=-1, vmax=1,
                ax=ax_heatmap,
                cbar_ax=ax_colorbar,
                cbar_kws={'label': 'Pearson Correlation Coefficient'})

    # 设置标签和标题
    ax_heatmap.set_xlabel('Raman Shift Features', fontsize=12)
    ax_heatmap.set_ylabel('Significantly Correlated Genes', fontsize=12)
    ax_heatmap.set_title(title, fontsize=16, pad=20)

    plt.tight_layout()
    return fig


# ==================== 4. 统计报告 ====================
def generate_statistics_report(correlation_df, significant_df, threshold):
    """生成统计分析报告"""
    n_total_genes = len(correlation_df['Gene'].unique())
    n_total_raman = len(correlation_df['Raman_Shift'].unique())
    n_significant_genes = len(significant_df['Gene'].unique()) if not significant_df.empty else 0
    n_significant_pairs = len(significant_df)

    print("=" * 60)
    print("Raman-Gene Correlation Analysis Report")
    print("=" * 60)
    print(f"Total Raman features: {n_total_raman}")
    print(f"Total genes analyzed: {n_total_genes}")
    print(f"Significant genes (|r| > {threshold}): {n_significant_genes}")
    print(f"Significant gene-raman pairs: {n_significant_pairs}")
    print(f"Correlation threshold: |r| > {threshold}")

    if not significant_df.empty:
        print(f"\nTop 10 strongest correlations:")
        top_10 = significant_df.reindex(significant_df['Pearson_r'].abs().sort_values(ascending=False).index).head(10)
        print(top_10.to_string(index=False))

        # 统计正负相关性
        positive_pairs = len(significant_df[significant_df['Pearson_r'] > 0])
        negative_pairs = len(significant_df[significant_df['Pearson_r'] < 0])
        print(f"\nPositive correlations: {positive_pairs}")
        print(f"Negative correlations: {negative_pairs}")


# ==================== 主程序 ====================
# ==================== 主程序 ====================
if __name__ == "__main__":
    # 文件路径
    gene_path = r"E:\Study\yjs\y1\data\组学与拉曼相关性\找差异基因与拉曼光谱\M2_gene.xlsx"
    raman_path = r"E:\Study\yjs\y1\data\组学与拉曼相关性\找差异基因与拉曼光谱\Raman_M2.xlsx"
    output_all_path = r"E:\Study\yjs\y1\data\组学与拉曼相关性\找差异基因与拉曼光谱\M2拉曼1440基因相关性.xlsx"

    # 指定的拉曼位移
    target_raman_shifts = ["1440"
        # # 第一组：644-662 cm⁻¹ (低波数区域)
        # "644", "645", "646", "647", "648", "650",
        # "651", "652", "653", "655", "656", "657",
        # "658", "659", "661", "662",
        #
        # # 第二组：1581-1600 cm⁻¹ (高波数区域，芳香环C=C伸缩振动特征区)
        # "1581", "1582", "1583", "1584", "1585", "1586", "1587",
        # "1588", "1589", "1590", "1591", "1592", "1593", "1594",
        # "1595", "1596", "1597", "1598", "1599", "1600"
    ]
    # 参数设置
    correlation_threshold = 0.8

    # 1. 数据加载
    print("正在加载数据...")
    df_raman, df_gene = load_and_preprocess_data(gene_path, raman_path)
    if df_raman is None:
        exit()

    print(f"拉曼数据形状: {df_raman.shape}")
    print(f"基因数据形状: {df_gene.shape}")

    # 2. 计算相关性
    print("正在计算相关性并筛选显著相关对...")
    corr_matrix, significant_pairs = calculate_correlation(df_raman[target_raman_shifts], df_gene, correlation_threshold)

    # # 3. 输出结果和统计
    # generate_statistics_report(corr_matrix, significant_pairs, correlation_threshold)

    # 4. 保存指定拉曼位移的相关性结果
    print(f"\n正在保存指定拉曼位移的相关性结果...")

    # 筛选出只包含目标拉曼位移的相关性数据
    filtered_correlation_matrix = corr_matrix

    # 保存筛选后的相关性数据
    filtered_correlation_matrix.to_excel(output_all_path, index=True)
    print(f"指定拉曼位移的基因相关性对已保存至: {output_all_path}")

    # 5. 可视化 - 只对显著相关的部分生成热图
    if not significant_pairs.empty:
        print("\n正在生成热图...")

        fig = plot_correlation_heatmap(
            corr_matrix,
            significant_pairs,
            title=f"Raman-Gene Correlation (|r| > {correlation_threshold})\n"
                  f"{len(significant_pairs['Gene'].unique())} genes correlated with {len(significant_pairs['Raman_Shift'].unique())} Raman shifts"
        )

        if fig:
            plt.savefig(r"E:\Study\yjs\y1\data\组学与拉曼相关性\找差异基因与拉曼光谱\correlation_heatmap(2).svg",
                        format="svg", dpi=300, bbox_inches='tight')
            plt.show()
    else:
        print("没有显著相关的数据，跳过可视化")

