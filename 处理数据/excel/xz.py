import numpy as np
import pandas as pd

# 设置显示精度
pd.set_option('display.float_format', '{:.10f}'.format)


# ==================== TOPSIS 核心函数 ====================
def topsis_score(data_matrix, weights):
    """
    TOPSIS计算相对贴合度得分
    data_matrix: 归一化后的矩阵 (n_samples, n_features)
    weights: 权重数组，长度等于特征数
    返回: 得分数组
    """
    # 加权规范化矩阵
    v = data_matrix * weights
    # 最优与最劣方案（指标均已正向化）
    v_best = np.max(v, axis=0)
    v_worst = np.min(v, axis=0)
    # 距离
    d_best = np.sqrt(np.sum((v - v_best) ** 2, axis=1))
    d_worst = np.sqrt(np.sum((v - v_worst) ** 2, axis=1))
    # 相对贴合度
    score = d_worst / (d_best + d_worst)
    return score


def normalize_matrix(X, neg_indices):
    """
    极值归一化，并将负向指标转为正向
    X: 原始数据 (n_samples, n_features)
    neg_indices: 负向指标列索引列表（0-based）
    返回: 归一化正向矩阵
    """
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
                # 负向指标：越大越差，采用逆向归一化
                X_norm[:, j] = (max_val - col) / (max_val - min_val)
            else:
                # 正向指标
                X_norm[:, j] = (col - min_val) / (max_val - min_val)
    return X_norm


# ==================== 主函数（用户自定义权重） ====================
def compute_with_custom_weights(df, weights, id_cols=4, total_indices=20, neg_tail=3):
    """
    使用自定义权重逐年计算TOPSIS得分
    参数:
        df: DataFrame，前id_cols列为标识，随后total_indices列为指标
        weights: list/array，长度等于total_indices，用户指定的权重（需归一化？内部会检查）
        id_cols: 标识列数量（默认4）
        total_indices: 指标总列数（默认20）
        neg_tail: 最后几列为负向指标（默认3）
    返回:
        score_df: 包含原始ID列和得分的DataFrame
    """
    # 权重转为numpy数组并归一化（保证和为1）
    w = np.array(weights, dtype=np.float64)
    if np.abs(np.sum(w) - 1.0) > 1e-10:
        print("警告：输入权重之和不为1，将自动归一化。")
        w = w / np.sum(w)

    # 提取数据
    id_df = df.iloc[:, :id_cols]
    data_df = df.iloc[:, id_cols:id_cols + total_indices]

    # 负向指标位置
    neg_indices = list(range(total_indices - neg_tail, total_indices))

    # 按年份分组
    years = id_df.iloc[:, 1].unique()  # 第2列为年份
    years = sorted(years)

    score_list = []
    for year in years:
        year_mask = id_df.iloc[:, 1] == year
        X_raw = data_df[year_mask].values.astype(np.float64)

        # 缺失值处理：列均值填充
        col_means = np.nanmean(X_raw, axis=0)
        inds = np.where(np.isnan(X_raw))
        X_raw[inds] = np.take(col_means, inds[1])

        # 归一化
        X_norm = normalize_matrix(X_raw, neg_indices)

        # TOPSIS计算得分
        scores = topsis_score(X_norm, w)
        scores = np.round(scores, 10)

        # 保存
        year_id = id_df[year_mask].iloc[:, :id_cols].copy()
        year_id['score'] = scores
        score_list.append(year_id)

    score_df = pd.concat(score_list, ignore_index=True)
    return score_df


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # ---------- 方式1：直接在代码中填写20个指标的权重 ----------
    # 请按照指标顺序填写权重（总和应为1，若不为一程序会自动归一化）
    custom_weights = [
        0.09776, 0.08696, 0.05115, 0.10398, 0.01606,  # 前5个指标
        0.05886, 0.03622, 0.04124, 0.01851, 0.04839,  # 第6-10个
        0.03775, 0.11405, 0.01807, 0.09645, 0.01421,  # 第11-15个
        0.03412, 0.1027, 0.0129, 0.00675, 0.00385  # 第16-20个（最后3个为负向指标）
    ]  # 这里假设等权重，您需要修改为实际值

    # ---------- 方式2：从Excel文件读取权重（可选） ----------
    # 如果权重保存在Excel中（例如一行20个数值），取消下面注释即可使用
    # weight_df = pd.read_excel("weights.xlsx", header=None)
    # custom_weights = weight_df.values.flatten().tolist()
    # 确保长度等于20，且顺序与数据中的指标顺序一致

    # ---------- 读取面板数据 ----------
    # 替换为您的数据文件路径
    df = pd.read_excel("新质生产力.xlsx")  # 或 pd.read_csv("data.csv")

    # 计算得分
    score_df = compute_with_custom_weights(
        df,
        weights=custom_weights,
        id_cols=4,
        total_indices=20,
        neg_tail=3
    )

    # 预览结果
    print("\n===== 各省得分示例（前10行）=====")
    print(score_df.head(10))

    # 保存结果
    with pd.ExcelWriter("自定义权重得分结果.xlsx", engine='openpyxl') as writer:
        score_df.to_excel(writer, sheet_name="得分表", index=False)
        # 同时保存所用权重供查阅
        weight_series = pd.Series(custom_weights, name="自定义权重")
        weight_series.to_excel(writer, sheet_name="权重表", index=True)

    print("\n结果已保存至：自定义权重得分结果.xlsx")