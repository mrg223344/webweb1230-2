import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import xgboost as xgb
import os

# 绘图字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif'] 
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="MUMPP转SMPP风险预测", layout="wide")

# ==========================================
# 1. 新版模型加载函数
# ==========================================
@st.cache_resource
def load_robust_model():
    # 检查文件是否存在
    if not os.path.exists("xgb_core_model.json") or not os.path.exists("feature_meta.pkl"):
        return None, None
    
    # 1. 加载特征名
    meta = joblib.load('feature_meta.pkl')
    feature_names = meta['feature_names']
    
    # 2. 初始化一个空模型架构 (参数要与训练时大致一致，但 weights 会被覆盖)
    model = xgb.XGBClassifier(base_score=0.5) 
    
    # 3. 加载 JSON 权重 (这是最安全的方式)
    model.load_model("xgb_core_model.json")
    
    return model, feature_names

# 执行加载
model, feature_names = load_robust_model()

if model is None:
    st.error("❌ 未找到模型文件。请先运行新的 train_model.py 生成 'xgb_core_model.json'。")
    st.stop()

# ==========================================
# 2. 侧边栏输入
# ==========================================
st.sidebar.header("📝 患者临床指标输入")
input_data = {}

for col in feature_names:
    if 'SEX' in col.upper():
        input_data[col] = st.sidebar.selectbox(f"{col} (性别)", [1, 2], format_func=lambda x: "男" if x==1 else "女")
    elif 'AGE' in col.upper():
        input_data[col] = st.sidebar.number_input(f"{col} (年龄)", min_value=1, max_value=120, value=50)
    else:
        input_data[col] = st.sidebar.number_input(f"输入 {col}", value=0.0)

input_df = pd.DataFrame([input_data], columns=feature_names)

# ==========================================
# 3. 预测与 SHAP
# ==========================================
st.title("🏥 MUMPP 进展为 SMPP 概率预测平台")
st.markdown("---")

col1, col2 = st.columns([1.5, 2.5])

with col1:
    st.subheader("📊 预测结果")
    
    if st.button("开始预测 (Predict)", type="primary"):
        # 1. 预测
        prediction_prob = model.predict_proba(input_df)[:, 1][0]
        
        risk_level = "高风险" if prediction_prob > 0.5 else "低风险"
        st.metric(label="SMPP 发生概率", value=f"{prediction_prob:.2%}", delta=risk_level)
        
        if prediction_prob > 0.5:
            st.error("⚠️ 提示：高风险患者。")
        else:
            st.success("✅ 提示：低风险患者。")

        # ==========================================
        # 4. SHAP (修复版)
        # ==========================================
        with col2:
            st.subheader("🔍 归因分析 (SHAP)")
            with st.spinner("正在计算..."):
                
                # 获取底层 Booster
                booster = model.get_booster()
                booster.feature_names = feature_names
                
                # 因为我们在训练时加了 base_score=0.5，这里的格式应该是正确的
                # 即使如此，为了防止万一，我们加一个 try-catch 保护
                try:
                    explainer = shap.TreeExplainer(booster)
                    shap_values = explainer(input_df)
                    
                    plt.figure(figsize=(10, 6))
                    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
                    st.pyplot(plt.gcf())
                    plt.clf()
                    
                except Exception as e:
                    st.error(f"SHAP 图生成失败: {e}")
                    st.write("建议检查 xgboost 版本: pip install xgboost==1.7.6")
            
    else:
        st.info("👈 请在左侧输入数据并点击预测")