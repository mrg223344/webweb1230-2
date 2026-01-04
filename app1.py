import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import xgboost
import json

# ==========================================
# 0. 环境配置与中文支持
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 页面设置
st.set_page_config(page_title="MUMPP转SMPP风险预测", layout="wide")

# ==========================================
# 1. 加载模型 (增加鲁棒性)
# ==========================================
@st.cache_resource
def load_model():
    try:
        # 使用 joblib 加载模型包
        package = joblib.load('xgb_smpp_model.pkl')
        # 兼容性检查：如果是字典格式则提取，否则直接返回
        if isinstance(package, dict):
            model = package['model']
            feature_names = package['feature_names']
        else:
            model = package
            feature_names = model.feature_names_in_.tolist()  # 尝试从 sklearn 包装器提取
        return model, feature_names
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None, None

model, feature_names = load_model()
if model is None:
    st.error("❌ 未找到有效模型。请确保 'xgb_smpp_model.pkl' 文件在当前目录下。")
    st.stop()

# ==========================================
# 2. 侧边栏输入 (添加单位并优化数值格式)
# ==========================================
st.sidebar.header("📝 患者临床指标输入")
input_data = {}
units = {}  # 可以自定义每个特征的单位，例如：units['AGE'] = '岁'，units['其他指标'] = 'mmol/L' 等
# 示例：假设非SEX/AGE的指标单位为 'mmol/L'，可根据实际调整

with st.sidebar:
    for col in feature_names:
        # 根据特征名关键词自动适配输入组件，并添加单位
        label = col
        if 'SEX' in col.upper():
            label += " (性别)"
            input_data[col] = st.selectbox(label, [1, 2], format_func=lambda x: "男" if x == 1 else "女")
        elif 'AGE' in col.upper():
            label += " (岁)"
            input_data[col] = st.number_input(label, min_value=1, max_value=120, value=50)
        else:
            # 假设其他指标的单位为 'mmol/L'，可根据实际修改
            unit = units.get(col, 'mmol/L')  # 默认单位
            label += f" ({unit})"
            input_data[col] = st.number_input(label, value=0.0, format="%.2f")  # 保留2位小数

# 转换为 DataFrame 并保持特征顺序
input_df = pd.DataFrame([input_data])[feature_names]

# ==========================================
# 3. 主界面预测逻辑
# ==========================================
st.title("🏥 MUMPP 进展为 SMPP 概率预测平台")
st.markdown("---")
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("📊 预测概率")
    if st.button("开始预测 (Predict)", type="primary"):
        # 1. 计算概率
        prediction_prob = model.predict_proba(input_df)[:, 1][0]
        
        # 2. 显示仪表盘
        risk_level = "高风险 (High Risk)" if prediction_prob > 0.5 else "低风险 (Low Risk)"
        st.metric(label="SMPP 发生概率", value=f"{prediction_prob:.2%}", delta=risk_level, delta_color="inverse")
        
        if prediction_prob > 0.5:
            st.error("⚠️ 该患者进展风险较高，建议密切关注。")
        else:
            st.success("✅ 该患者目前风险较低。")

        # ==========================================
        # 4. SHAP 解释 (优化错误处理)
        # ==========================================
        with col2:
            st.subheader("🔍 归因分析 (SHAP Explanation)")
            try:
                with st.spinner("计算特征贡献度中..."):
                    # 使用 shap.Explainer 作为首选，兼容性更好
                    explainer = shap.Explainer(model)
                    shap_values = explainer(input_df)
                    # 绘图
                    fig, ax = plt.subplots(figsize=(10, 6))
                    # 绘制瀑布图
                    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
                    # 调整布局防止标签重叠
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.clf()
                
                st.info("💡 解释：**红色**条形表示增加风险的因素，**蓝色**条形表示降低风险的因素。")
            
            except Exception as e:
                st.warning(f"SHAP 绘图尝试中... (错误: {e})")
                # Fallback: 手动处理 Booster 逻辑
                try:
                    booster = model.get_booster()
                    booster.feature_names = feature_names
                    explainer = shap.TreeExplainer(booster)
                    shap_values = explainer.shap_values(input_df)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.clf()
                except Exception as fallback_e:
                    st.error(f"SHAP 解释失败: {fallback_e}")
    else:
        st.info("👈 请在左侧输入临床数据，然后点击按钮获取预测结果。")

st.markdown("---")
st.caption("注：本系统基于机器学习模型生成预测，仅供辅助科研参考，不作为临床诊断唯一依据。")
