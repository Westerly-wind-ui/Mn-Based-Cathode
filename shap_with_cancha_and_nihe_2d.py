import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 定义保存路径
output_folder = r"D:\matlab工具箱\代码\循环稳定性-最新\SHAP 2d"
# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 加载数据
X = np.load(r'X_py_all2.npy')
feature_names = ['NE', 'Na', 'Ni', 'Fe', 'AE', 'IE', 'IEC', 'SD', 'V', 'AX', 'CX', 'V-MIN', 'V-MAX', 'PS']
X_py = pd.DataFrame(X, columns=feature_names)

# 加载SHAP值
shap_values = np.load(r'shap_values_all.npy')
shap_values_df = pd.DataFrame(shap_values, columns=feature_names)

def plot_shap_interaction(X_py, shap_values_df):
    # 生成所有特征对
    feature_pairs = [(i, j) for i in range(len(feature_names)) for j in range(len(feature_names)) if i < j]
    for i, j in feature_pairs:
        feature_i = feature_names[i]
        feature_j = feature_names[j]

        # 同步清洗数据
        mask = (
            X_py[feature_i].notna() &
            X_py[feature_j].notna() &
            shap_values_df[feature_i].notna() &
            shap_values_df[feature_j].notna()
        )
        feature_i_clean = X_py.loc[mask, feature_i]
        feature_j_clean = X_py.loc[mask, feature_j]
        shap_i_clean = shap_values_df.loc[mask, feature_i]
        shap_j_clean = shap_values_df.loc[mask, feature_j]

        # 计算交互SHAP值
        interactive_shap = shap_i_clean * shap_j_clean

        if len(interactive_shap) < 1:
            print(f"Skipping {feature_i}-{feature_j}: 无有效数据")
            continue

        # 创建画布
        plt.figure(figsize=(10, 6), dpi=150)
        ax = plt.gca()

        # 绘制一维散点
        scatter = ax.scatter(
            x=feature_i_clean,
            y=interactive_shap,  # 使用SHAP交互值作为纵坐标
            c=feature_j_clean,  # 用另一个特征值着色
            cmap='coolwarm',
            s=40,
            alpha=0.7,
            edgecolor='w',
            linewidth=0.5,
            zorder=2
        )

        # 添加颜色条
        cbar = plt.colorbar(scatter, orientation='vertical', pad=0.02)
        cbar.set_label(f'{feature_j} Value', rotation=270, labelpad=15, fontsize=14)

        # 添加分布密度曲线
        sns.kdeplot(
            x=feature_i_clean,
            y=interactive_shap,
            color='darkblue',
            fill=True,
            alpha=0.3,
            ax=ax,
            zorder=1
        )

        # 坐标轴设置
        ax.set_xlabel(f'{feature_i} Value', fontsize=16)
        ax.set_ylabel('SHAP Interaction Value', fontsize=16)
        ax.set_title(f'2D SHAP Interaction: {feature_i} vs. {feature_j}', pad=10, fontsize=16)

        # 设置坐标轴刻度字体大小
        ax.tick_params(axis='both', which='major', labelsize=20)

        # 网格线优化
        ax.grid(True, linestyle='--', alpha=0.6)
        sns.despine()

        # 保存输出
        output_path = os.path.join(output_folder, f'shap_interaction_{feature_i}_{feature_j}.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Generated {output_path} (Points: {len(interactive_shap)})")

plot_shap_interaction(X_py, shap_values_df)
