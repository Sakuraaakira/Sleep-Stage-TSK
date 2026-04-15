import numpy as np


def solve_beta_dp(H, T, Y_prev, lambda_p, delta_p):
    """
    引入类别权重惩罚，防止模型只预测 S2 和 Wake
    """
    N, L = H.shape
    num_classes = T.shape[1]

    # --- 新增：计算类别权重 ---
    # 计算每个类别的样本数
    class_counts = np.sum(T, axis=0)
    # 计算权重：样本越少，权重越高
    weights = N / (num_classes * (class_counts + 1e-6))
    # 构建权重矩阵 W [N, N]
    # 取每个样本对应类别的权重
    sample_weights = T @ weights
    W_mat = np.diag(sample_weights)

    # 岭回归正则化项
    I = np.eye(L) * 1e-4

    # 使用加权最小二乘公式：beta = (H^T * W * H + I)^-1 * H^T * W * Target
    if Y_prev is None:
        return np.linalg.inv(H.T @ W_mat @ H + I) @ H.T @ W_mat @ T
    else:
        target = T - (1 - delta_p) * Y_prev
        scaled_H = (1 + lambda_p) * H
        # 加权求解 Equation (11)
        return np.linalg.inv(scaled_H.T @ W_mat @ scaled_H + I) @ scaled_H.T @ W_mat @ target


def solve_final_lambdas(all_outputs, T, xi, zeta):
    """严格执行 Equation (18): 线性组合权重求解"""
    DP = len(all_outputs)
    num_classes = T.shape[1]
    lambdas = np.zeros((DP, num_classes))

    for c in range(num_classes):
        # 构建组合矩阵 Fi [N, DP]
        Fi_c = np.column_stack([out[:, c] for out in all_outputs])
        target_c = T[:, c]

        # 求解 lambda
        inv_term = np.linalg.inv(Fi_c.T @ Fi_c + np.eye(DP) * 1e-5)
        # Eq. 18 解析解
        lambdas[:, c] = (1.0 / (1.0 + zeta)) * inv_term @ (Fi_c.T @ ((xi + zeta) * target_c))

    return lambdas


def random_projection_layer(Y, d_target):
    """
    Section III-B: 随机投影增强逻辑
    用于将上一层的输出投影并拼接到下一层的输入中
    """
    rs = np.random.RandomState(42)  # 固定种子保证复现
    # 随机投影矩阵 (对应你提到的随机偏置/增强矩阵)
    W = rs.randn(Y.shape[1], d_target)
    return Y @ W