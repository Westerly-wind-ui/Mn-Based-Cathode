import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import metrics
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def calculate_residuals(y, y_pred, dataset_name):
    residuals = y - y_pred
    residuals_df = pd.DataFrame({
        'True Value': y,
        'Predicted Value': y_pred,
        'Residual': residuals,
        'Data Set': dataset_name
    })
    return residuals_df

def plot_residual_distribution(i, x_train, x_pred_train, x_test, x_pred_test, x_val, x_pred_val):
    # 创建训练集、测试集和验证集 DataFrame
    data_train = pd.DataFrame({'True Value': x_train, 'Predicted Value': x_pred_train, 'Data Set': 'Train'})
    data_test = pd.DataFrame({'True Value': x_test, 'Predicted Value': x_pred_test, 'Data Set': 'Test'})
    data_val = pd.DataFrame({'True Value': x_val, 'Predicted Value': x_pred_val, 'Data Set': 'Validation'})

    # 合并所有 DataFrame
    data = pd.concat([data_train, data_test, data_val])

    # 计算 R^2 分数
    r2_train = metrics.r2_score(x_train, x_pred_train)
    r2_test = metrics.r2_score(x_test, x_pred_test)
    r2_val = metrics.r2_score(x_val, x_pred_val)

    # 设置调色板
    palette = ["C20", "#A184BC", "#ff7f0e"]

    # 设置图形大小
    plt.figure(figsize=(12, 10), dpi=1200)

    # 创建 JointGrid 对象
    g = sns.JointGrid(data=data, x="True Value", y="Predicted Value", height=10)
    # 在主图中绘制散点图，区分训练集和测试集
    g.plot_joint(sns.scatterplot, hue='Data Set', data=data, palette=palette, alpha=0.6, s = 200)
    # 分别绘制两个数据集的边缘密度图，不显示图例
    sns.kdeplot(data=data, x='True Value', hue='Data Set', ax=g.ax_marg_x, fill=True, common_norm=False,
                palette=palette, alpha=0.5, legend=False)
    sns.kdeplot(data=data, y='Predicted Value', hue='Data Set', ax=g.ax_marg_y, fill=True, common_norm=False,
                palette=palette, alpha=0.5, legend=False)
    # 添加 y=x 对角线
    g.ax_joint.plot([data['True Value'].min(), data['True Value'].max()],
                    [data['True Value'].min(), data['True Value'].max()], color='red', linestyle='--',
                    label='y=x')
    # 设置标签和标题
    g.set_axis_labels('True Values', 'Predicted Values', fontsize=16)  # 增加字体大小
    # 调整横纵坐标数字的字体大小
    g.ax_joint.tick_params(axis='both', which='major', labelsize=20)  # 增加刻度标签的字体大小
    # 仅在主图中显示图例
    g.ax_joint.legend(title='Dataset', loc='upper left', fontsize=16)  # 增加字体大小
    # 在右下角添加箱线图，分别显示训练集和测试集的残差分布
    ax_inset = inset_axes(g.ax_joint, width="40%", height="20%", loc='lower right',
                          bbox_to_anchor=(0.2, 0.05, 0.8, 0.8), bbox_transform=g.ax_joint.transAxes)
    # 绘制横向箱线图，区分训练集和测试集的残差
    train_residuals = calculate_residuals(x_train, x_pred_train, "Train")
    test_residuals = calculate_residuals(x_test, x_pred_test, "Test")
    val_residuals = calculate_residuals(x_val, x_pred_val, "Validation")
    sns.boxplot(data=train_residuals, y='Data Set', x='Residual', palette=palette, ax=ax_inset)
    sns.boxplot(data=test_residuals, y='Data Set', x='Residual', palette=palette, ax=ax_inset)
    sns.boxplot(data=val_residuals, y='Data Set', x='Residual', palette=palette, ax=ax_inset)
    ax_inset.set_title('Residuals', fontsize=16)  # 增加字体大小
    ax_inset.set_xlabel('', fontsize=16)  # 增加字体大小
    ax_inset.set_ylabel('', fontsize=16)  # 增加字体大小
    ax_inset.yaxis.set_visible(False)
    # 添加 R^2 文本
    g.ax_joint.text(0.95, 0.37, f'Train $R^2$ = {r2_train:.3f}', transform=g.ax_joint.transAxes, fontsize=14,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
    g.ax_joint.text(0.95, 0.32, f'Test $R^2$ = {r2_test:.3f}', transform=g.ax_joint.transAxes, fontsize=14, verticalalignment='bottom',
                    horizontalalignment='right', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
    g.ax_joint.text(0.95, 0.27, f'Validation $R^2$ = {r2_val:.3f}', transform=g.ax_joint.transAxes, fontsize=14,
                    verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
    # 调整整体边距，将图向右移动
    plt.subplots_adjust(left=0.1)

    # 保存图像并增加边界留白
    plt.savefig(f"residuals_plot_{i}.png", format='png', bbox_inches='tight', pad_inches=0.5)
    plt.show()

if __name__ == '__main__':
    # 包含训练集、验证集和测试集
    y_train = np.load(r"train_y_feature_label_py.npy")
    y_pred_train = np.load(r"y_train_pred_all_py.npy")
    y_test = np.load(r"test_y_feature_label_py.npy")
    y_pred_test = np.load(r"y_test_pred_all_py.npy")
    y_val = np.load(r"vaild_y_feature_label_py.npy")
    y_pred_val = np.load(r"y_vaild_pred_all_py.npy")

    # 调用函数并传递 i 的值
    plot_residual_distribution(1, y_train, y_pred_train, y_test, y_pred_test, y_val, y_pred_val)