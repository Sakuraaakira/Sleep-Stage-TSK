import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis
import pywt
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler


def extract_paper_features(epochs, fs=250, n_components=10):
    """丰富特征集：频域比例 + 时域统计 + Hjorth参数 + 小波能量"""
    all_features = []
    for epoch in epochs:
        f_vec = []
        # 1. 时域
        f_vec.extend([np.mean(epoch), np.std(epoch), skew(epoch), kurtosis(epoch)])
        # 2. Hjorth 参数
        act = np.var(epoch)
        mob = np.sqrt(np.var(np.diff(epoch)) / (act + 1e-9))
        f_vec.extend([act, mob])
        # 3. 频域 (5个睡眠频段比例)
        freqs, psd = welch(epoch, fs=fs, nperseg=fs * 2)
        total_p = np.sum(psd) + 1e-9
        bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 12), 'sigma': (12, 16), 'beta': (16, 30)}
        for b in bands.values():
            f_vec.append(np.sum(psd[(freqs >= b[0]) & (freqs <= b[1])]) / total_p)
        # 4. DB6 小波各层能量
        coeffs = pywt.wavedec(epoch, 'db6', level=4)
        for c in coeffs:
            f_vec.append(np.sum(np.square(c)) / (len(c) + 1e-9))

        all_features.append(f_vec)

    X = np.array(all_features)
    X_scaled = StandardScaler().fit_transform(X)
    # KPCA 降维至 10 维
    kpca = KernelPCA(n_components=n_components, kernel='rbf', gamma=0.1)
    return kpca.fit_transform(X_scaled)
