import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import os

def ale_plot_stylish_with_vertical_distribution(npy_file, feature="X_1", bins=10, mc_samples=50):
    """
    从 npy 文件中加载 ALE 数据并绘图：
      1) 绘制蒙特卡洛曲线（低透明度蓝色）和主 ALE 曲线（黑色实线，无点）；
      2) 添加 rugplot（灰色小竖线）表示原始特征的分布；
      3) 在 x 轴下方用蓝色竖线（色号 "#1f77b4"）表示原始特征的分布，
         竖线密度反映数据密集程度。
    """
    # --- 样式设置 ---
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["axes.linewidth"] = 1.2
    plt.rcParams["axes.edgecolor"] = "0.4"
    plt.rcParams["grid.color"] = "0.8"
    plt.rcParams["grid.linestyle"] = "-"
    plt.rcParams["grid.linewidth"] = 0.6

    # --- 数据加载 ---
    data = np.load(npy_file, allow_pickle=True).item()
    bin_centers = data["bin_centers"]
    ale_effect = data["ale_effect"]
    accumulated_effect = data["accumulated_effect"]
    mc_ales = data.get("mc_ales", None)  # shape: (num_bins, mc_samples)
    train_feature = data.get("train_feature", None)

    # --- 绘图 ---
    fig, ax = plt.subplots(figsize=(8, 6))

    # 绘制蒙特卡洛曲线（低透明度蓝色）
    if mc_ales is not None:
        for i in range(mc_ales.shape[1]):
            ax.plot(bin_centers, mc_ales[:, i],
                    color="#1f77b4", alpha=0.1, linewidth=1.0)

    # 绘制主 ALE 曲线（黑色无点实线）
    ax.plot(bin_centers, ale_effect, color="black",
            linewidth=2.5, label="ALE Curve")

    # 添加 rugplot（灰色小竖线），显示原始特征分布
    if train_feature is not None and len(train_feature) > 0:
        sns.rugplot(train_feature, ax=ax, color="#1f77b4", alpha=0.2)

    # 设置坐标标签、标题、网格等
    ax.set_xlabel(f"Feature '{feature}'")
    ax.set_ylabel("Accumulated Local Effect")
    title_main = f"First-order ALE of feature '{feature}'"
    title_sub = f"Bins: {bins} - Monte-Carlo Samples: {mc_samples}"
    ax.set_title("\n".join((title_main, title_sub)), pad=12)
    ax.grid(True, linestyle="-", alpha=0.4)

    # 动态调整 y 轴范围：考虑 ALE 和蒙特卡洛曲线的整体范围
    y_min_all = np.min(ale_effect)
    y_max_all = np.max(ale_effect)
    if mc_ales is not None:
        y_min_all = min(y_min_all, np.min(mc_ales))
        y_max_all = max(y_max_all, np.max(mc_ales))
    padding = 0.1 * (y_max_all - y_min_all) if (y_max_all > y_min_all) else 1
    ax.set_ylim(y_min_all - padding, y_max_all + padding)



    ax.legend(loc="best")
    plt.tight_layout()
    return ax


if __name__ == "__main__":
    # 指定包含 npy 文件的文件夹路径
    folder_path = r"D:\matlab工具箱\代码\循环稳定性-最新"

    # 遍历文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        # 检查文件是否以 ale_data_ 开头并以 .npy 结尾
        if file_name.startswith("ale_data_") and file_name.endswith(".npy"):
            # 提取特征名
            feature = file_name.split("ale_data_")[1].split(".npy")[0]
            file_path = os.path.join(folder_path, file_name)

            # 生成图表
            ax = ale_plot_stylish_with_vertical_distribution(file_path, feature=feature, bins=10, mc_samples=50)

            # 保存图表到文件
            output_file = os.path.join(folder_path, f"ale_plot_{feature}.png")
            plt.savefig(output_file)
