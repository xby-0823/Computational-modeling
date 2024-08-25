import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, norm
from scipy.special import comb
from tqdm import tqdm


# 从Excel表格读取数据
df_1 = pd.read_excel("GCC_loss.xlsx", names=["gcc loss%"])  # 读取Excel中的gcc loss
df_2 = df_1.dropna(axis=0, how='any')  # 剔除None值
gcc_loss = df_2.values
all_patients = len(gcc_loss)  # 获取患者数量

# 参数设置
# rescue_rate = float(input("请输入rescue rate: "))
# n = int(input("请输入重复次数n: "))
# m = int(input("请输入每组样本数m: "))
rescue_rate = 0.7
m = 132
n = 10000

alpha = 0.05
effective_clinical_trials = 0
clinical_power_total = 0

bar = tqdm(range(n))
for _ in bar:
    # 从N位患者中随机选取2m个样本
    selected_patients = np.random.choice(all_patients, 2 * m, replace=False)

    # 计算RGC loss
    selected_gcc_loss = gcc_loss[selected_patients]
    rgc_loss = 2.697 * selected_gcc_loss - 2.445

    # 计算RGC rescue
    RGC_rescue_control = rgc_loss[:m]                       # treatment_group
    RGC_rescue_treatment = rescue_rate * rgc_loss[m:]       # control_group

    # 计算均值和方差
    x_bar_control = np.mean(RGC_rescue_control)
    x_bar_treatment = np.mean(RGC_rescue_treatment)
    s_squared_control = np.var(RGC_rescue_control, ddof=0)
    s_squared_treatment = np.var(RGC_rescue_treatment, ddof=0)

    # 计算beta、p-value和clinical power
    x_bar = 1.645 * np.sqrt(s_squared_control / m) + x_bar_control        # 计算临界值x_bar
    z = (x_bar - x_bar_treatment) / np.sqrt(s_squared_treatment / m)
    beta = 1 - norm.cdf(z)                 # 查表

    t, p = ttest_ind(RGC_rescue_control, RGC_rescue_treatment)
    clinical_power = 1 - beta

    # 判断假设是否通过
    if p < alpha:
        if x_bar_treatment < x_bar_control:
            effective_clinical_trials += 1

    clinical_power_total += clinical_power

# 计算clinical power的均值
average_clinical_power = clinical_power_total / n

# 输出结果
print("总样本数:", all_patients)
print("clinical power:", average_clinical_power)
print("effective clinical trials:", effective_clinical_trials)

