import numpy as np
from sklearn.cluster import KMeans
from drrh_utils import solve_beta_dp, solve_final_lambdas


class DrrhTSKFC:
    def __init__(self, config):
        self.cfg = config
        self.sub_clfs = []
        self.lambdas = None
        # 严格遵循论文：预存随机投影矩阵 W (Fig. 1 中的随机权重)
        self.W_projs = [np.random.RandomState(i).randn(6, 6) for i in range(config.DP)]

    def _get_h_matrix(self, X, centers, sigmas, masks):
        """Characteristic 2: 短规则逻辑 (Short Rules)"""
        N, L = X.shape[0], centers.shape[0]
        H = np.zeros((N, L))
        for l in range(L):
            # 应用 Mask，使规则只关注部分特征
            diff = (X - centers[l]) * masks[l]
            s = sigmas[l] + 1e-9
            dist = np.sum((diff ** 2) / (2 * s ** 2), axis=1)
            H[:, l] = np.exp(-dist)
        return H / (np.sum(H, axis=1, keepdims=True) + 1e-9)

    def train(self, X_train, T_train):
        N, D_orig = X_train.shape
        Y_prev, X_current = None, X_train
        all_sub_outputs = []

        for dp in range(self.cfg.DP):
            L = self.cfg.L
            D_curr = X_current.shape[1]

            # 短规则掩码 (10% 比例)
            masks = np.ones((L, D_curr))
            n_short = int(L * self.cfg.SHORT_RULE_RATIO)
            for i in range(n_short):
                m_idx = np.random.choice(D_curr, D_curr // 2, replace=False)
                masks[i, m_idx] = 0

            # 聚类初始化中心
            kmeans = KMeans(n_clusters=L, n_init=5, random_state=dp).fit(X_current)
            centers = kmeans.cluster_centers_
            sigmas = np.ones_like(centers) * (np.std(X_current, axis=0) + 0.1) * 0.7

            # Characteristic 1: 随机规则继承 (Rule Heritage)
            heritage_mask = np.zeros(L, dtype=bool)
            if dp > 0:
                n_heritage = int(L * self.cfg.HERITAGE_RATIO)
                prev_p = self.sub_clfs[dp - 1]
                centers[:n_heritage, :prev_p['centers'].shape[1]] = prev_p['centers'][:n_heritage]
                heritage_mask[:n_heritage] = True

            # 求解本层参数 (Equation 11)
            H = self._get_h_matrix(X_current, centers, sigmas, masks)
            beta = solve_beta_dp(H, T_train, Y_prev, self.cfg.LAMBDA_PRIME, self.cfg.DELTA_PRIME)

            Y_dp = H @ beta
            all_sub_outputs.append(Y_dp)
            self.sub_clfs.append({
                'beta': beta, 'centers': centers, 'sigmas': sigmas,
                'masks': masks, 'dp': dp + 1, 'heritage_mask': heritage_mask
            })

            # 准备下一层输入 (Original X + Projection of current output)
            X_current = np.hstack([X_train, Y_dp @ self.W_projs[dp]])
            Y_prev = Y_dp

        # 最终线性组合权重求解 (Equation 18)
        self.lambdas = solve_final_lambdas(all_sub_outputs, T_train, self.cfg.XI, self.cfg.ZETA)

    def predict(self, X_test):
        """严格按照论文推理流程：逐层前传 + 最终组合"""
        all_sub_outputs = []
        X_current = X_test
        for dp in range(self.cfg.DP):
            p = self.sub_clfs[dp]
            H = self._get_h_matrix(X_current, p['centers'], p['sigmas'], p['masks'])
            Y_dp = H @ p['beta']
            all_sub_outputs.append(Y_dp)
            # 更新下一层输入
            X_current = np.hstack([X_test, Y_dp @ self.W_projs[dp]])

        # 最终组合 Equation 14
        final_scores = np.zeros_like(all_sub_outputs[0])
        for j in range(self.cfg.DP):
            final_scores += all_sub_outputs[j] * self.lambdas[j]
        return np.argmax(final_scores, axis=1)

    def predict_raw(self, X_test):
        """获取 Equation (14) 计算出的 6 维得分向量矩阵 [N, 6]"""
        all_sub_outputs, X_current = [], X_test
        for dp in range(self.cfg.DP):
            p = self.sub_clfs[dp]
            H = self._get_h_matrix(X_current, p['centers'], p['sigmas'], p['masks'])
            Y_dp = H @ p['beta']
            all_sub_outputs.append(Y_dp)
            # 投影并更新输入
            X_current = np.hstack([X_test, Y_dp @ self.W_projs[dp]])

        final_scores = np.zeros_like(all_sub_outputs[0])
        for j in range(self.cfg.DP):
            final_scores += all_sub_outputs[j] * self.lambdas[j]
        return final_scores

    def display_rules(self):
        """严格对照论文 Table III 的语言解释展示"""
        print("\n" + "=" * 90)
        print("TABLE III: LINGUISTIC EXPLANATIONS OF FUZZY RULES (PAPER REPLICATION)")
        print("=" * 90)

        def get_term(val):
            v = np.clip(val, 0, 1)
            if v < 0.2:
                return "not serious"
            elif v < 0.4:
                return "little serious"
            elif v < 0.6:
                return "medium serious"
            elif v < 0.8:
                return "more serious"
            else:
                return "especially serious"

        for clf in self.sub_clfs:
            dp = clf['dp']
            print(f"\n[Sub-classifier DP={dp}]")
            for r in [0, self.cfg.L - 1]:
                heritage = " (Inherited)" if clf['heritage_mask'][r] else " (New)"
                short = " (Short Rule)" if np.sum(clf['masks'][r]) < clf['centers'].shape[1] else ""
                c, b = clf['centers'][r], clf['beta'][r]
                terms = f"B1: {get_term(c[0]):15} | B2: {get_term(c[1]):15} | B3: {get_term(c[2]):15}"
                print(f"  Rule {r + 1}{heritage}{short}: IF {terms} -> THEN Weights[:2]: [{b[0]:.3f}, {b[1]:.3f}]")

        print("\n" + "-" * 90)
        print(f"Final Combined Lambda Weights: {[round(float(np.mean(l)), 4) for l in self.lambdas]}")
        print("=" * 90)