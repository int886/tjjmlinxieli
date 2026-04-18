import os
import pandas as pd
from pyreadstat import write_sav

# 你的文件夹路径
folder_path = r'D:\tjjmlinxieli\处理数据\excel\低空经济测度指标'


# 自动清洗成SPSS合法变量名
def clean_var_name(name):
    name = str(name).strip()
    name = name.replace(" ", "").replace("\n", "").replace("\t", "")
    # 数字开头 → 前面加V
    if name[0].isdigit():
        name = "V" + name
    # 只保留字母数字下划线
    clean = ""
    for c in name:
        if c.isalnum() or c == "_":
            clean += c
    # 不能为空
    return clean if clean else "V"


for filename in os.listdir(folder_path):
    if filename.lower().endswith(".xlsx"):
        excel_file = os.path.join(folder_path, filename)
        df = pd.read_excel(excel_file, sheet_name=0)

        # 批量清洗所有列名
        df.columns = [clean_var_name(col) for col in df.columns]

        sav_name = os.path.splitext(filename)[0] + ".sav"
        sav_path = os.path.join(folder_path, sav_name)
        write_sav(df, sav_path)
        print(f"✅ 转换成功：{filename} → {sav_name}")