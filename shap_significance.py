import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests

# 加载数据
shap_values = np.load(r'shap_interaction_values_py_all.npy')
print("SHAP interaction values shape:", shap_values.shape)
n_samples, n_features, _ = shap_values.shape

# 特征名称
feature_names = ['NE', 'Na', 'Ni', 'Fe', 'AE', 'IE', 'IEC', 'SD', 'V', 'AX', 'CX', 'V-MIN', 'V-MAX', 'PS']

# 初始化 p 值矩阵
p_values = np.zeros((n_features, n_features))

# 遍历所有特征对，计算 p 值
for i in range(n_features):
    for j in range(i + 1, n_features):
        interaction_values = shap_values[:, i, j]
        t_stat, p_val = stats.ttest_1samp(interaction_values, 0)
        p_values[i, j] = p_val
        print(f"特征 {feature_names[i]} 和特征 {feature_names[j]} 的交互，p值为: {p_val}")

# 提取上三角的 p 值
p_values_flat = p_values[np.triu_indices(n_features, k=1)]

# 多重检验校正
corrected_p_values = multipletests(p_values_flat, method='bonferroni')[1]

# 将校正后的 p 值填充回矩阵
p_values[np.triu_indices(n_features, k=1)] = corrected_p_values

# 创建一个掩码，隐藏下三角部分
mask = np.tril(np.ones((n_features, n_features)), k=-1).astype(bool)

# 使用 viridis 颜色映射，并设置颜色范围
cmap = 'viridis'  # 或者使用 'plasma'
vmin, vmax = -1.5, 2.5  # 根据提供的图设置颜色范围
# 设置注释字体大小
annot_kws = {"size": 32}  # 调整为合适的字体大小

# 绘制热图
plt.figure(figsize=(10, 8))
sns.heatmap(p_values, annot=True, cmap=cmap, xticklabels=feature_names, yticklabels=feature_names,
            mask=mask, cbar_kws={'shrink': 0.8, 'label': 'S/C'}, vmin=vmin, vmax=vmax)

# 设置图表标题和标签
plt.title('Corrected p-value matrix', fontsize=16)
plt.xlabel('Feature', fontsize=16)
plt.ylabel('Feature', fontsize=16)
plt.xticks(rotation=45, fontsize=14)
plt.yticks(rotation=45, fontsize=14)
plt.tight_layout()
plt.show()

# 筛选显著的特征对
threshold = 0.05
significant_pairs = []
for i in range(n_features):
    for j in range(i + 1, n_features):
        if p_values[i, j] < threshold:
            significant_pairs.append((feature_names[i], feature_names[j], p_values[i, j]))

# 输出显著的特征对
print("\n显著的特征对 (p < 0.05):")
for pair in significant_pairs:
    print(f"特征 {pair[0]} 和特征 {pair[1]} 的交互，p值为: {pair[2]:.4f}")

# 绘制显著特征对的柱状图
if significant_pairs:
    labels = [f"{pair[0]} & {pair[1]}" for pair in significant_pairs]
    p_vals = [pair[2] for pair in significant_pairs]

    plt.figure(figsize=(10, 6))
    plt.barh(labels, p_vals, color='skyblue')
    plt.title('Significant Interaction P-values (p < 0.05)', fontsize=16)
    plt.xlabel('P-value', fontsize=16)
    plt.ylabel('Feature Pair', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16, rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("\n没有特征对的 p 值小于 0.05")

    # 注意
    # print出的 p 值和热图上显示的 p 值确实存在不一致的情况。这可能是由于以下原因之一：
    # 多重检验校正：
    # 在代码中，你对 p 值进行了多重检验校正（如 Bonferroni 校正）。校正后的 p 值可能与未校正的 p 值差异较大。
    # 热图上显示的是校正后的 p 值，而打印的 p 值是未校正的原始 p 值。