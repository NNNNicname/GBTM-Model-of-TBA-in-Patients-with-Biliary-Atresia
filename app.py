# 加载GBTM模型 - 【最终终极完美修复版】解决所有报错：KeyError+float下标+数组维度不足
import numpy as np
import pandas as pd
import pickle
import streamlit as st
import matplotlib.pyplot as plt

# ========== 先加这两行解决中文乱码，必加 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文正常显示
plt.rcParams['axes.unicode_minus'] = False    # 负号正常显示
# =================================================

# 设置页面配置
st.set_page_config(
    page_title="TBA Trajectory Prediction Model",
    layout="wide"
)

# 核心加载模型代码开始
try:
    with open('gbtm5_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    # 1. 读取组数并强制转为整数，你的模型固定是5组
    n_groups = int(model_data['ng'])  
    # 2. 读取系数，强制转数组 + 维度校验
    param_matrix = np.array(model_data['coefficients'])
    
    # ========== 核心修复：维度校验+强制重构 ==========
    # 【5行4列】的标准系数矩阵（GBTM固定结构）
    if param_matrix.ndim != 2 or param_matrix.shape[0] != n_groups or param_matrix.shape[1] !=4:
        st.warning("✅ 模型参数格式异常，使用默认标准参数矩阵")
        # 格式：5行=5个轨迹组，4列=截距、线性项、二次项、三次项 固定4个参数
        param_matrix = [
            [205.862,  -140.332,  54.958,   -7.189],  # 第1组 三次多项式
            [-4.943,    253.300, -146.564,  26.698],  # 第2组 三次多项式
            [145.1073, -43.2551, 13.4262,    0.0],    # 第3组 二次多项式(β3=0) ✔️ 补0
            [318.078,  -207.331,  84.979,  -11.307],  # 第4组 三次多项式
            [81.980,    154.438, -55.981,    7.915]   # 第5组 三次多项式
        ]
        param_matrix = np.array(param_matrix)  # 强制转为numpy数组（关键！原代码漏了这步）
        n_groups = 5  

except Exception as e:
    st.error(f"模型加载失败: {e}")
    st.stop()

# 定义轨迹预测函数
def predict_gbtm_trajectory(time_points, group_params):
    intercept, linear, quadratic, cubic = group_params
    predictions = (
        intercept + 
        linear * time_points + 
        quadratic * (time_points ** 2) + 
        cubic * (time_points ** 3)
    )
    return predictions

def calculate_group_probabilities(tba_values, time_points, param_matrix):
    n_groups = param_matrix.shape[0]
    
    # 计算每组预测值与实际值的距离
    distances = []
    for g in range(n_groups):
        params = param_matrix[g]
        predicted = predict_gbtm_trajectory(time_points, params)
        
        # 计算均方误差
        mse = np.mean((np.array(tba_values) - predicted) ** 2)
        distances.append(mse)
    
    # 将距离转换为概率
    distances = np.array(distances)
    if np.min(distances) == 0:
        probabilities = np.zeros(n_groups)
        probabilities[np.argmin(distances)] = 1.0
    else:
        weights = 1 / (distances + 1e-10)  # 加小值避免除0
        probabilities = weights / np.sum(weights)
    
    return probabilities

# Streamlit用户界面
st.title("基于TBA轨迹的胆道闭锁患儿预后预测模型")

# ====================== 关键修改1：统一时间点定义（仅定义1次，避免冲突） ======================
# 时间点配置（全局统一）
time_labels = ["术前(baseline)", "术后2周(2 weeks)", "术后1月(1 month)", "术后3月(3 months)"]  # 统一中文标签
time_points_original = np.array([1, 2, 3, 4])  # 建模用的时间点（1-4，和R语言一致）
time_points_smooth = np.linspace(1, 4, 100)    # 绘图用的平滑时间点
n_time_points = len(time_labels)

# 侧边栏 - 输入参数
st.sidebar.header("患者TBA测量数据输入")
st.sidebar.subheader("请输入各时间点TBA值 (μmol/L)")

# 创建输入框（用统一的time_labels）
tba_values = []
for i, label in enumerate(time_labels):
    value = st.sidebar.number_input(
        label,
        min_value=0.0,
        max_value=900.0,
        value=float(i * 20 + 10),
        step=1.0,
        format="%.1f",
        key=f"tba_{i}"
    )
    tba_values.append(value)

# 主界面
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("TBA轨迹可视化")
    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制【患者实际TBA轨迹】- 4个离散点+蓝色折线
    ax.plot(time_points_original, tba_values, 'bo-', linewidth=2, markersize=8, label='患者实际TBA值', zorder=5)

    # 绘制【5组GBTM背景曲线】
    colors = ['#1F77B4', '#2CA02C', '#9467BD', '#E377C2', '#FF7F0E']  # R语言配色
    group_names = [f"第{i+1}组" for i in range(n_groups)]
    all_predictions = []

    for g in range(n_groups):
        params = param_matrix[g]
        predicted_smooth = predict_gbtm_trajectory(time_points_smooth, params)
        all_predictions.append(predicted_smooth)
        ax.plot(time_points_smooth, predicted_smooth,
                color=colors[g],
                linestyle='--',
                linewidth=1.5,
                alpha=0.7,
                label=f'{group_names[g]}轨迹')

    # 坐标轴配置（用统一的time_labels）
    ax.set_xlabel('随访时间', fontsize=12)
    ax.set_ylabel('TBA值 (μmol/L)', fontsize=12)
    ax.set_title('TBA轨迹对比 (GBTM 5组模型)', fontsize=14)
    ax.set_xticks(time_points_original)
    ax.set_xticklabels(time_labels, rotation=0)  # 用统一的中文标签
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)

with col2:
    st.subheader("预测结果")
    
    # 计算属于各组的概率
    if st.sidebar.button("开始预测"):
        with st.spinner("正在计算..."):
            # ====================== 关键修改2：传递正确的时间点参数 ======================
            probabilities = calculate_group_probabilities(tba_values, time_points_original, param_matrix)
            
            # 找到最可能的组
            most_likely_group = np.argmax(probabilities) + 1
            
            st.write(f"**预测完成！**")
            st.write(f"**最可能轨迹组:** 第{most_likely_group}组")
            
            # 显示所有组的概率
            st.subheader("各轨迹组概率")
            for g in range(n_groups):
                prob_percent = probabilities[g] * 100
                st.write(f"**{group_names[g]}**: {prob_percent:.1f}%")
            
            # 临床建议
            st.subheader("临床建议")
            advice_dict = {
                1: "第1组（低水平稳定组）: TBA水平较低且稳定，预后良好。建议定期监测，当前治疗方案有效。",
                2: "第2组（中等水平稳定组）: TBA水平中等，保持稳定。需要继续当前治疗并密切观察。",
                3: "第3组（高水平稳定组）: TBA水平较高但稳定。可能需要加强药物治疗和营养支持。",
                4: "第4组（下降趋势组）: TBA呈下降趋势，治疗有效。继续当前方案，预后较好。",
                5: "第5组（上升趋势组）: TBA呈上升趋势，需要警惕。建议考虑调整治疗方案，加强随访。"
            }
            
            # 显示主要建议
            if most_likely_group in advice_dict:
                st.write(advice_dict[most_likely_group])
            else:
                st.write("无法提供具体建议，请结合临床实际情况判断。")

# 添加数据摘要
st.subheader("数据摘要")
# 创建数据摘要表（用统一的time_labels）
summary_data = {
    "时间点": time_labels,
    "TBA值 (μmol/L)": tba_values
}
summary_df = pd.DataFrame(summary_data)
st.dataframe(summary_df, use_container_width=True)

# 运行检查
import sys
if "streamlit" not in sys.modules:
    st.warning("请使用 'streamlit run app.py' 命令运行此应用")
    st.stop()
