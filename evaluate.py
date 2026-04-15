import numpy as np
from sklearn.metrics import confusion_matrix


def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    acc = np.sum(np.diag(cm)) / np.sum(cm)

    sens, specs = [], []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fp + fn)
        sens.append(tp / (tp + fn + 1e-9))
        specs.append(tn / (tn + fp + 1e-9))

    # 返回论文中使用的术语
    return {
        "ACC": acc * 100,
        "SEN": np.mean(sens) * 100,
        "SPE": np.mean(specs) * 100
    }