import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess  # 新增导入


# 定义保存路径
output_folder = r"D:\matlab工具箱\代码\已发表论文（处理完成）\Github代码\SHAP 1d"
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


def plot_true_1d_shap(X_py, shap_values_df):
    for feature in feature_names:
        # 同步清洗数据
        mask = (
            X_py[feature].notna() &
            shap_values_df[feature].notna()
        )
        feature_clean = X_py.loc[mask, feature].astype(float)
        shap_clean = shap_values_df.loc[mask, feature].astype(float)

        if len(shap_clean) < 1:
            print(f"Skipping {feature}: 无有效数据")
            continue

        plt.figure(figsize=(10, 6), dpi=150)
        ax = plt.gca()

        # 绘制一维散点
        scatter = ax.scatter(
            x=feature_clean,
            y=shap_clean,
            c=feature_clean,
            cmap='coolwarm',
            s=60,
            alpha=0.7,
            edgecolor='w',
            linewidth=0.5,
            zorder=2
        )

        # 添加颜色条
        cbar = plt.colorbar(scatter, orientation='vertical', pad=0.02)
        cbar.set_label(f'{feature} Value', rotation=270, labelpad=15, fontsize=14)

        # 添加分布密度曲线
        sns.kdeplot(
            x=feature_clean,
            y=shap_clean,
            color='darkblue',
            fill=True,
            alpha=0.3,
            ax=ax,
            zorder=1
        )

        # 添加 LOWESS 趋势线
        lowess_smoothed = lowess(shap_clean, feature_clean, frac=0.3, it=0)
        ax.plot(
            lowess_smoothed[:, 0],
            lowess_smoothed[:, 1],
            color='#5E3C99',
            linewidth=2.5,
            label='Trend (LOWESS)',
            zorder=3
        )

        # 添加图例
        ax.legend(fontsize=12)

        # 坐标轴设置
        ax.set_xlabel(f'{feature} Value', fontsize=16)
        ax.set_ylabel('SHAP Value', fontsize=16)
        ax.set_title(f'1D SHAP Distribution: {feature}', pad=10, fontsize=16)

        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.grid(True, linestyle='--', alpha=0.6)
        sns.despine()

        output_path = os.path.join(output_folder, f'True_1D_{feature}.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved: {output_path} (Points: {len(shap_clean)})")


plot_true_1d_shap(X_py, shap_values_df)