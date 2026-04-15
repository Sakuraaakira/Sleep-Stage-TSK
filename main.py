import numpy as np
from sklearn.preprocessing import OneHotEncoder
from config import Config
from drrh_model import DrrhTSKFC
from data_loader import load_mit_bih_data
from preprocessing import extract_paper_features
from evaluate import calculate_metrics


def main():
    print("Step 1: Loading all 18 MIT-BIH Records...")
    data_dir = r"E:\jsjsj\data"
    all_X, all_y = [], []
    records = ['slp01a', 'slp01b', 'slp02a', 'slp02b', 'slp03', 'slp04', 'slp14', 'slp16', 'slp32', 'slp37', 'slp41',
               'slp45', 'slp48', 'slp59', 'slp60', 'slp61', 'slp66', 'slp67x']

    for rec in records:
        try:
            epochs, labels, fs = load_mit_bih_data(data_dir, rec)
            if len(epochs) > 0:
                X_f = extract_paper_features(epochs, fs=fs, n_components=Config.N_COMPONENTS)
                all_X.append(X_f);
                all_y.append(labels)
                print(f"  Loaded {rec}")
        except:
            pass

    X, y = np.vstack(all_X), np.concatenate(all_y)
    enc = OneHotEncoder(sparse_output=False)
    T = enc.fit_transform(y.reshape(-1, 1))

    indices = np.random.permutation(len(X))
    split = int(len(X) * Config.TRAIN_SIZE)
    X_train, X_test = X[indices[:split]], X[indices[split:]]
    T_train, T_test = T[indices[:split]], T[indices[split:]]
    y_test = y[indices[split:]]

    print(f"\nStep 2: Training Drrh-TSK-FC on {len(X_train)} samples...")
    model = DrrhTSKFC(Config)
    model.train(X_train, T_train)

    print("\n" + "#" * 65)
    print("FIGURE 5: TRACS/TEACS (%) OF Drrh-TSK-FC AT EACH LAYER (DP)")
    print("#" * 65)

    X_tr_curr, X_te_curr = X_train, X_test
    for j in range(Config.DP):
        clf = model.sub_clfs[j]
        H_tr = model._get_h_matrix(X_tr_curr, clf['centers'], clf['sigmas'], clf['masks'])
        trac = np.mean(np.argmax(H_tr @ clf['beta'], axis=1) == np.argmax(T_train, axis=1)) * 100
        H_te = model._get_h_matrix(X_te_curr, clf['centers'], clf['sigmas'], clf['masks'])
        teac = np.mean(np.argmax(H_te @ clf['beta'], axis=1) == y_test) * 100
        print(f"DP = {j + 1} | Trac: {trac:.2f}% | Teac: {teac:.2f}%")

        # 为下一层手动更新输入
        X_tr_curr = np.hstack([X_train, (H_tr @ clf['beta']) @ model.W_projs[j]])
        X_te_curr = np.hstack([X_test, (H_te @ clf['beta']) @ model.W_projs[j]])

    print("-" * 65)
    # 调用 predict 函数
    y_pred = model.predict(X_test)
    m = calculate_metrics(y_test, y_pred)
    print(f"FINAL COMBINED RESULTS: ACC: {m['ACC']:.2f}% | SEN: {m['SEN']:.2f}% | SPE: {m['SPE']:.2f}%")
    print("#" * 65)

    model.display_rules()


if __name__ == "__main__":
    main()