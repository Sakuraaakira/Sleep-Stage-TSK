import numpy as np
import matplotlib.pyplot as plt
import wfdb
import os
from scipy.signal import welch, butter, filtfilt, detrend

# ================== 1. 环境配置 ==================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def clean_eeg_signal(data, fs):
    """深度清洁脑电信号：去趋势 + 0.5-30Hz带通 + 异常值抑制"""
    data = detrend(data)
    # 抑制强心电干扰：削掉超过3倍标准差的尖峰
    std = np.std(data)
    data = np.clip(data, -3 * std, 3 * std)
    # 滤波
    b, a = butter_bandpass(0.5, 30.0, fs)
    data = filtfilt(b, a, data)
    return (data - np.mean(data)) / (np.std(data) + 1e-9)


def plot_feature_extraction_analysis():
    print("正在进行全阶段特征提取分析...")

    data_dir = r"E:\jsjsj\data"
    record_name = 'slp16'  # 强烈建议换成 slp03，信号比 slp01a 纯净10倍，更容易看清特征

    from data_loader import load_mit_bih_data
    epochs_all, labels_all, fs = load_mit_bih_data(data_dir, record_name)

    # 睡眠阶段映射
    stage_map = {0: 'Wake', 1: 'S1', 2: 'S2', 3: 'S3', 4: 'S4', 5: 'REM'}
    colors = ['#1f77b4', '#9467bd', '#ff7f0e', '#d62728', '#8c564b', '#2ca02c']

    # 自动获取所有存在的阶段
    present_stages = sorted(np.unique(labels_all))

    fig = plt.figure(figsize=(16, 12))
    plt.suptitle(f"受试者 {record_name}：全睡眠阶段多维度特征对比分析", fontsize=18, y=0.98)

    # --- 图 1：时域形态对比 (Time Domain) ---
    ax1 = plt.subplot(3, 1, 1)
    offset = 0
    for s_val in present_stages:
        idx = np.where(labels_all == s_val)[0][len(np.where(labels_all == s_val)[0]) // 2]
        clean_sig = clean_eeg_signal(epochs_all[idx], fs)
        t = np.linspace(0, 30, len(clean_sig))
        ax1.plot(t, clean_sig + offset, label=stage_map[s_val], color=colors[s_val], linewidth=0.8)
        offset -= 4  # 纵向错开

    ax1.set_title("1. 时域特征：各阶段脑电波形态 (已去除干扰并纵向偏移展示)", loc='left', fontsize=14)
    ax1.set_xlabel("时间 (秒)");
    ax1.set_yticks([]);
    ax1.legend(loc='upper right', ncol=len(present_stages))

    # --- 图 2：频域能量密度 (Frequency Domain) ---
    ax2 = plt.subplot(3, 1, 2)
    # 标注生理频段
    bands = {'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 'Sigma': (13, 16), 'Beta': (16, 30)}
    for name, (low, high) in bands.items():
        ax2.axvspan(low, high, alpha=0.05, label=f'_{name}')
        ax2.text((low + high) / 2, 0.085, name, ha='center', fontsize=10, color='gray')

    for s_val in present_stages:
        idx_list = np.where(labels_all == s_val)[0]
        # 对该阶段所有样本求平均功率谱，结果更稳健
        all_psds = []
        for i in idx_list[:10]:  # 取前10个样本求平均
            f, p = welch(clean_eeg_signal(epochs_all[i], fs), fs, nperseg=fs * 4)
            all_psds.append(p / np.sum(p))  # 面积归一化

        avg_p = np.mean(all_psds, axis=0)
        ax2.plot(f, avg_p, label=f'{stage_map[s_val]}', color=colors[s_val], linewidth=2.5)

    ax2.set_title("2. 频域特征：平均功率谱密度 (PSD) - 展示不同阶段的能量重心偏移", loc='left', fontsize=14)
    ax2.set_xlim(0.5, 25);
    ax2.set_ylim(0, 0.1);
    ax2.set_ylabel("相对功率");
    ax2.legend()

    # --- 图 3：非线性特征对比 (Complexity) ---
    ax3 = plt.subplot(3, 1, 3)
    comp_vals = []
    for s_val in present_stages:
        idx_list = np.where(labels_all == s_val)[0]
        c = [np.std(np.diff(clean_eeg_signal(epochs_all[i], fs))) for i in idx_list[:20]]
        comp_vals.append(np.mean(c))

    ax3.bar([stage_map[s] for s in present_stages], comp_vals, color=[colors[s] for s in present_stages], width=0.5)
    ax3.set_title("3. 非线性特征：Hjorth 复杂度平均值 (反映脑电信号的随机性与活跃度)", loc='left', fontsize=14)
    ax3.set_ylabel("复杂度系数")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    plot_feature_extraction_analysis()