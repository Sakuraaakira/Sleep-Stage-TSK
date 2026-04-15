import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import KernelPCA
from config import Config
from drrh_model import DrrhTSKFC
from data_loader import load_mit_bih_data
from preprocessing import extract_paper_features


def save_everything():
    print("开始在本地进行全量训练...")
    data_dir = r"E:\jsjsj\data"  # 这里用你本地的路径
    all_X_raw = []  # 存储未降维的特征
    all_y = []
    records = ['slp01a', 'slp01b', 'slp02a', 'slp02b', 'slp03', 'slp04', 'slp14', 'slp16', 'slp32', 'slp37', 'slp41',
               'slp45', 'slp48', 'slp59', 'slp60', 'slp61', 'slp66', 'slp67x']

    # 1. 提取所有人的基础特征（不在这里做 KPCA，因为 KPCA 要统一做）
    # 注意：这里需要稍微改一下 preprocessing 的逻辑，先拿到全量高维特征
    from scipy.signal import welch
    from scipy.stats import skew, kurtosis
    import pywt

    def get_raw_feat(epochs, fs=250):
        feats = []
        for epoch in epochs:
            f = [np.mean(epoch), np.std(epoch), skew(epoch), kurtosis(epoch)]
            act = np.var(epoch);
            mob = np.sqrt(np.var(np.diff(epoch)) / (act + 1e-9))
            f.extend([act, mob])
            freqs, psd = welch(epoch, fs=fs, nperseg=fs * 2)
            tp = np.sum(psd) + 1e-9
            bands = {'d': (0.5, 4), 't': (4, 8), 'a': (8, 12), 's': (12, 16), 'b': (16, 30)}
            for b in bands.values(): f.append(np.sum(psd[(freqs >= b[0]) & (freqs <= b[1])]) / tp)
            for c in pywt.wavedec(epoch, 'db6', level=4): f.append(np.sum(np.square(c)) / (len(c) + 1e-9))
            feats.append(f)
        return feats

    for rec in records:
        try:
            e, l, fs = load_mit_bih_data(data_dir, rec)
            if len(e) > 0:
                all_X_raw.extend(get_raw_feat(e, fs))
                all_y.extend(l)
                print(f"已处理 {rec}")
        except:
            pass

    X_raw = np.array(all_X_raw)
    y = np.array(all_y)

    # 2. 统一做标准化和 KPCA 并保存这两个“转换器”
    print("正在进行特征降维...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    kpca = KernelPCA(n_components=Config.N_COMPONENTS, kernel='rbf', gamma=0.1)
    X_reduced = kpca.fit_transform(X_scaled)

    # 3. 训练模型
    print("正在训练 Drrh-TSK-FC 模型...")
    T = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))
    model = DrrhTSKFC(Config)
    model.train(X_reduced, T)

    # 4. 把模型、标准化器、KPCA 压成一个包
    trained_package = {
        'model': model,
        'scaler': scaler,
        'kpca': kpca
    }

    with open('trained_sleep_model.pkl', 'wb') as f:
        pickle.dump(trained_package, f)

    print("成功！已生成 trained_sleep_model.pkl，大小约几MB。")


if __name__ == "__main__":
    save_everything()