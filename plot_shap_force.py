import matplotlib

matplotlib.use('Agg')
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


def shap_force_plot(shap_values, base_value, sample_data, feature_names, save_path):
    # 设置全局字体大小
    plt.rcParams.update({'font.size': 14})  # 增大字体
    plt.figure(figsize=(12, 6))  # 调整画布尺寸
    plt.tight_layout(pad=4)  # 调整边距

    # 创建Explanation对象
    explainer = shap.Explanation(
        values=np.round(shap_values, 2),
        base_values=np.round(base_value, 2),
        data=np.round(sample_data, 2),
        feature_names=feature_names
    )

    # 生成力图
    plt.figure(figsize=(12, 6), dpi=300)  # 调整画布尺寸和分辨率
    shap.plots.force(explainer, matplotlib=True, show=False)

    # 获取当前图中的所有标注
    ax = plt.gca()
    texts = ax.texts

    # 手动调整 f(x) 和 base value 的位置
    for text in texts:
        if text.get_text() == "f(x)":  # 找到 f(x) 标注
            current_x, current_y = text.get_position()
            new_y = current_y - 0.8  # 向下移动更多
            text.set_position((current_x, new_y))
            text.set_bbox(dict(facecolor='none', edgecolor='none', boxstyle='square'))
        elif text.get_text() == "base value":  # 找到 base value 标注
            current_x, current_y = text.get_position()
            new_y = current_y - 0.8  # 向下移动更多
            text.set_position((current_x, new_y))
            text.set_bbox(dict(facecolor='none', edgecolor='none', boxstyle='square'))

    # 手动设置坐标轴范围防止截断
    ax = plt.gca()
    total_effect = np.sum(explainer.values)
    ax.set_xlim(left=explainer.base_values - 0.5, right=explainer.base_values + total_effect + 0.5)

    # 调整布局并保存
    plt.tight_layout(pad=2)  # 调整边距
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


if __name__ == '__main__':
    # 加载数据（示例路径）
    test_data = np.load(r'D:\matlab工具箱\代码\循环稳定性-最新\X_py_all2.npy')
    base_value = np.load(r'D:\matlab工具箱\代码\循环稳定性-最新\expected_value_all_py.npy').item()
    shap_values = np.load(r'D:\matlab工具箱\代码\循环稳定性-最新\shap_values_all.npy')

    # 参数设置
    feature_names = ['NE', 'Na', 'Ni', 'Fe', 'AE', 'IE', 'IEC', 'SD', 'V', 'AX', 'CX', 'V-MIN', 'V-MAX', 'PS']
    sample_idx = 109 # 第一个样本
    sample_data = test_data[sample_idx]

    # 格式化数据
    sample_shap = np.round(shap_values[sample_idx], 2) if len(shap_values.shape) == 2 else np.round(shap_values, 2)
    sample_data = np.round(sample_data, 2)
    base_value = np.round(base_value, 2)

    # 生成图像的保存路径
    save_path = fr'D:\matlab工具箱\代码\循环稳定性-最新\SHAP heatmap\shap_force_sample_{sample_idx}.png'

    # 生成并保存图像
    shap_force_plot(sample_shap, base_value, sample_data, feature_names, save_path)

    print(f"Sample {sample_idx} processed and saved to {save_path}")