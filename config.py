class Config:
    DP = 4  # 深度：4层
    L = 12  # 规则数
    HERITAGE_RATIO = 0.2  # 20% 规则继承 (论文 Remark 1)
    SHORT_RULE_RATIO = 0.1  # 10% 短规则 (论文 Remark 1)

    LAMBDA_PRIME = 0.01  # Eq 11 参数
    DELTA_PRIME = 0.01  # Eq 11 参数
    XI = 0.5
    ZETA = 0.5

    N_COMPONENTS = 10
    FS = 250
    TRAIN_SIZE = 0.75