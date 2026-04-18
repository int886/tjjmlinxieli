import numpy as np
import pandas as pd
from scipy.linalg import eig

# 设置显示精度
pd.set_option('display.float_format', '{:.10f}'.format)


# ==================== AHP权重计算函数 ====================
def ahp_weights_from_importance(values):

    g = np.array(values, dtype=np.float64)
    m = len(g)

    # 1. 计算最大差异性系数比 D
    D = np.max(g) / np.min(g)

    # 2. 确定调整系数 α
    if D <= 9:
        alpha = int(np.round(D))
    else:
        alpha = 9

    # 3. 映射比率 R
    R = (D / alpha) ** (1 / (alpha - 1))

    # 4. 1-9标度映射表
    scales = np.array([R ** i for i in range(9)], dtype=np.float64)

    # 5. 构造判断矩阵
    A = np.zeros((m, m), dtype=np.float64)
    for j in range(m):
        for k in range(m):
            r_jk = g[j] / g[k]
            diff = np.abs(scales - r_jk)
            closest_idx = np.argmin(diff)
            A[j, k] = scales[closest_idx]

    # 6. 特征向量法求权重
    eigvals, eigvecs = eig(A)
    max_idx = np.argmax(np.real(eigvals))
    w = np.real(eigvecs[:, max_idx])
    w = w / np.sum(w)

    return w, R


# TOPSIS
def topsis_score(data_matrix, weights):
    """TOPSIS计算相对贴合度得分"""
    v = data_matrix * weights
    v_best = np.max(v, axis=0)
    v_worst = np.min(v, axis=0)
    d_best = np.sqrt(np.sum((v - v_best) ** 2, axis=1))
    d_worst = np.sqrt(np.sum((v - v_worst) ** 2, axis=1))
    score = d_worst / (d_best + d_worst)
    return score


def normalize_matrix(X, neg_indices):
    """极值归一化，负向指标转为正向"""
    n, m = X.shape
    X_norm = np.zeros_like(X, dtype=np.float64)
    for j in range(m):
        col = X[:, j]
        min_val = np.min(col)
        max_val = np.max(col)
        if max_val == min_val:
            X_norm[:, j] = 0.0
        else:
            if j in neg_indices:
                X_norm[:, j] = (max_val - col) / (max_val - min_val)
            else:
                X_norm[:, j] = (col - min_val) / (max_val - min_val)
    return X_norm


# ==================== 主函数 ====================
def compute_with_custom_importance(df, importance_values, id_cols=4, total_indices=20, neg_tail=3):

    # 1. 通过AHP映射法求最终权重
    final_weights, R = ahp_weights_from_importance(importance_values)
    final_weights = np.round(final_weights, 10)

    # 2. 提取数据
    id_df = df.iloc[:, :id_cols]
    data_df = df.iloc[:, id_cols:id_cols + total_indices]
    indicator_names = data_df.columns.tolist()
    neg_indices = list(range(total_indices - neg_tail, total_indices))

    # 3. 逐年计算得分
    years = id_df.iloc[:, 1].unique()
    years = sorted(years)
    score_list = []

    for year in years:
        year_mask = id_df.iloc[:, 1] == year
        X_raw = data_df[year_mask].values.astype(np.float64)

        # 归一化
        X_norm = normalize_matrix(X_raw, neg_indices)

        # TOPSIS得分
        scores = topsis_score(X_norm, final_weights)
        scores = np.round(scores, 10)

        # 保存
        year_id = id_df[year_mask].iloc[:, :id_cols].copy()
        year_id['score'] = scores
        score_list.append(year_id)

    score_df = pd.concat(score_list, ignore_index=True)
    final_weights_series = pd.Series(final_weights, index=indicator_names, name='AHP调整后权重')

    return score_df, final_weights_series


# 主体
if __name__ == "__main__":
    # 填写借助spsspro得到的20个指标的熵权法权重
    my_importance = [
        0.09776, 0.08696, 0.05115, 0.10398, 0.01606,
        0.05886, 0.03622, 0.04124, 0.01851, 0.04839,
        0.03775, 0.11405, 0.01807, 0.09645, 0.01421,
        0.03412, 0.1027, 0.0129, 0.00675, 0.00385
    ]

    #  读取面板数据
    df = pd.read_excel("新质生产力.xlsx")

    # 计算
    score_df, weights_series = compute_with_custom_importance(
        df,
        importance_values=my_importance,
        id_cols=4,
        total_indices=20,
        neg_tail=3
    )

    # 保存到Excel
    with pd.ExcelWriter("新质生产力测度结果.xlsx", engine='openpyxl') as writer:
        score_df.to_excel(writer, sheet_name="得分表", index=False)
        weights_df = pd.DataFrame(weights_series)
        weights_df.to_excel(writer, sheet_name="权重表", index=True)

    print("\n结果已保存至：新质生产力测度结果.xlsx")