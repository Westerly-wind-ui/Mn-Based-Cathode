import numpy as np
import shap
import matplotlib.pyplot as plt
import pandas as pd
import os

folder_path = 'D:/matlab工具箱/代码/循环稳定性-最新/SHAP'
os.makedirs(folder_path, exist_ok=True)

def plot_shap_summary(i,X_py,shap_values):
    # 确保 i 是整数类型
    i = int(i)
    #
    # # 加载数据
    # X_py = np.load(r'X_py_all.npy')
    # shap_values = np.load(r'H:shap_values_all.npy')

    # SHAP值特征贡献的蜂巢图
    plt.figure()
    shap.summary_plot(shap_values, X_py, show=False)
    # 拼接文件名并保存图像
    # plt.savefig(pic)
    pic = os.path.join(folder_path, f'SHAP_numpy summary_plot_'+str(i)+'.png')
    plt.savefig(pic, format='png', bbox_inches='tight')
    print(f"SHAP summary plot saved as '{1}'")

    # SHAP值排序的特征重要性柱状图
    plt.figure(figsize=(10, 5), dpi=1200)
    shap.summary_plot(shap_values, X_py, plot_type="bar", show=False)
    plt.title('SHAP_numpy Sorted Feature Importance')
    plt.tight_layout()
    pic1 = os.path.join(folder_path, f'SHAP_numpy Sorted Feature Importance with SHAP summary plot_' + str(i) + '.png')
    plt.savefig(pic1, format='png', bbox_inches='tight')


    # 结合蜂巢图与特征重要性图的双轴SHAP可视化图
    # 创建主图（用来画蜂巢图）
    fig, ax1 = plt.subplots(figsize=(10, 8), dpi=1200)
    # 在主图上绘制蜂巢图，并保留热度条
    #shap.summary_plot(shap_values, X_py, feature_names=X_py.columns, plot_type="dot", show=False, color_bar=True)
    shap.summary_plot(shap_values, X_py, feature_names=X_py.columns,plot_type="dot", show=False, color_bar=True)
    plt.gca().set_position([0.2, 0.2, 0.65, 0.65])  # 调整图表位置，留出右侧空间放热度条
    ax1 = plt.gca()
    ax2 = ax1.twiny()
    shap.summary_plot(shap_values, X_py, plot_type="bar", show=False)
    plt.gca().set_position([0.2, 0.2, 0.65, 0.65])  # 调整图表位置，与蜂巢图对齐
    ax2.axhline(y=13, color='gray', linestyle='-', linewidth=1)  # 注意y值应该对应顶部
    bars = ax2.patches  # 获取所有的柱状图对象
    for bar in bars:
        bar.set_alpha(0.2)  # 设置透明度
    pic2 = os.path.join(folder_path, f'SHAP_numpy Sorted Feature Importance_' + str(i) + '.png')
    ax1.set_xlabel('Shapley Value Contribution (Bee Swarm)', fontsize=16)
    ax2.set_xlabel('Mean Shapley Value (Feature Importance)', fontsize=16)
    ax2.xaxis.set_label_position('top')  # 将标签移动到顶部
    ax2.xaxis.tick_top()  # 将刻度也移动到顶部
    ax1.set_ylabel('Features', fontsize=16)
    plt.tight_layout()
    plt.savefig(pic2, format='png', bbox_inches='tight')


def plot_shap_summary_all(X_py,shap_values):
    # 确保 i 是整数类型
    # # i = int(i)
    #
    # # 加载数据
    # X_py = np.load(r'X_py_all.npy')
    # shap_values = np.load(r'shap_values_all.npy')

    # SHAP值特征贡献的蜂巢图
    plt.figure()
    shap.summary_plot(shap_values, X_py, show=False)
    # 拼接文件名并保存图像
    # plt.savefig(pic)
    pic = os.path.join(folder_path,f'SHAP_numpy summary_plot_all.png')
    plt.savefig(pic, format='png', bbox_inches='tight',dpi=600)
    print(f"SHAP summary plot saved as '{1}'")

    # SHAP值排序的特征重要性柱状图
    plt.figure(figsize=(10, 5), dpi=1200)
    shap.summary_plot(shap_values, X_py, plot_type="bar", show=False)
    plt.title('SHAP_numpy Sorted Feature Importance')
    plt.tight_layout()
    pic1 = os.path.join(folder_path, f'SHAP_numpy Sorted Feature Importance with SHAP summary plot_all.png')
    plt.savefig(pic1, format='png', bbox_inches='tight',dpi=600)


    # 结合蜂巢图与特征重要性图的双轴SHAP可视化图
    # 创建主图（用来画蜂巢图）
    fig, ax1 = plt.subplots(figsize=(10, 8), dpi=1200)
    # 在主图上绘制蜂巢图，并保留热度条
    #shap.summary_plot(shap_values, X_py, feature_names=X_py.columns, plot_type="dot", show=False, color_bar=True)
    shap.summary_plot(shap_values, X_py, feature_names=X_py.columns,plot_type="dot", show=False, color_bar=True)
    plt.gca().set_position([0.2, 0.2, 0.65, 0.65])  # 调整图表位置，留出右侧空间放热度条
    ax1 = plt.gca()
    ax2 = ax1.twiny()
    shap.summary_plot(shap_values, X_py, plot_type="bar", show=False)
    plt.gca().set_position([0.2, 0.2, 0.65, 0.65])  # 调整图表位置，与蜂巢图对齐
    ax2.axhline(y=13, color='gray', linestyle='-', linewidth=1)  # 注意y值应该对应顶部
    bars = ax2.patches  # 获取所有的柱状图对象
    for bar in bars:
        bar.set_alpha(0.2)  # 设置透明度
    pic2 = os.path.join(folder_path, f'SHAP_numpy Sorted Feature Importance_all.png')
    ax1.set_xlabel('Shapley Value Contribution (Bee Swarm)', fontsize=16)
    ax2.set_xlabel('Mean Shapley Value (Feature Importance)', fontsize=16)
    ax2.xaxis.set_label_position('top')  # 将标签移动到顶部
    ax2.xaxis.tick_top()  # 将刻度也移动到顶部
    ax1.set_ylabel('Features', fontsize=16)
    plt.tight_layout()
    plt.savefig(pic2, format='png', bbox_inches='tight',dpi=600)

def plot_shap_interaction(X_py,shap_values):
    plt.figure()
    shap.summary_plot(shap_values, X_py, show=False, max_display=6)
    pic_name = os.path.join(folder_path, f'shap_dependence_{feature_names}.png')

    # 保存图像
    plt.savefig(pic_name, format='png', bbox_inches='tight', dpi=600)
    print(f"SHAP_numpy interaction_all saved as '{1}'")


if __name__ == '__main__':

    #所有
     X = np.load(r'X_py_all2.npy')
     print (X)
     feature_names = ['NE', 'Na', 'Ni', 'Fe', 'AE', 'IE', 'IEC', 'SD', 'V', 'AX', 'CX', 'V-MIN', 'V-MAX', 'PS']
     X_py = pd.DataFrame(X, columns=feature_names)
     shap_values = np.load(r'shap_values_all.npy')
     plot_shap_summary_all(X_py, shap_values)
     shap_df = pd.DataFrame(shap_values,
                       columns=feature_names)  # 计算每个特征的平均 SHAP 值（绝对值），用于排序
     mean_shap_values = shap_df.abs().mean().sort_values(ascending=False) # 输出每个特征和对应的 SHAP 值
     print(mean_shap_values) # 如果你想把这些信息保存到文件中（例如 CSV 文件）
     mean_shap_values.to_csv('shap_feature_importance.csv')
     shap_values = np.load(r'shap_interaction_values_py_all.npy')
     print(shap_values)
     plot_shap_interaction(X_py, shap_values)

