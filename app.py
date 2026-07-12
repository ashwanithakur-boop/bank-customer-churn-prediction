import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
import json, os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Churn Risk Calculator",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1B4F72 0%, #2980B9 100%);
        color: white; padding: 28px 32px; border-radius: 10px;
        margin-bottom: 24px;
    }
    .main-header h1 { color: white; font-size: 28px; margin: 0; }
    .main-header p  { color: #D6EAF8; margin: 6px 0 0; font-size: 15px; }
    .metric-card {
        background: white; border-radius: 10px; padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center;
        border-left: 4px solid #2980B9;
    }
    .metric-card .val { font-size: 32px; font-weight: 700; }
    .metric-card .lbl { font-size: 13px; color: #7F8C8D; margin-top: 4px; }
    .risk-high   { border-left-color: #C0392B !important; }
    .risk-medium { border-left-color: #D4AC0D !important; }
    .risk-low    { border-left-color: #1E8449 !important; }
    .section-title {
        font-size: 17px; font-weight: 700; color: #1B4F72;
        border-bottom: 2px solid #2980B9; padding-bottom: 6px; margin: 18px 0 14px;
    }
    .insight-box {
        background: #EBF5FB; border-left: 4px solid #2980B9;
        padding: 14px 16px; border-radius: 0 8px 8px 0; margin: 10px 0;
        font-size: 14px; color: #1A5276;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model artifacts ──────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)

@st.cache_resource
def load_artifacts():
    model    = pickle.load(open(os.path.join(BASE,'model.pkl'),'rb'))
    features = pickle.load(open(os.path.join(BASE,'features.pkl'),'rb'))
    results  = json.load(open(os.path.join(BASE,'results.json')))
    return model, features, results

model, FEATURES, results = load_artifacts()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🏦 Bank Customer Churn Risk Intelligence System</h1>
  <p>Predictive modelling for proactive customer retention | Powered by Gradient Boosting ML</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯 Churn Risk Calculator", "📊 Model Performance", "📈 Data Insights"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Risk Calculator
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">Enter Customer Details</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📋 Demographics**")
        age         = st.slider("Age", 18, 92, 38)
        gender      = st.selectbox("Gender", ["Male","Female"])
        geography   = st.selectbox("Country", ["France","Germany","Spain"])

    with col2:
        st.markdown("**💰 Financial Profile**")
        credit_score = st.slider("Credit Score", 300, 850, 650)
        balance      = st.number_input("Account Balance (€)", 0, 300000, 60000, step=5000)
        salary       = st.number_input("Estimated Annual Salary (€)", 10000, 250000, 90000, step=5000)

    with col3:
        st.markdown("**🏦 Banking Behaviour**")
        tenure       = st.slider("Years with Bank (Tenure)", 0, 10, 4)
        num_products = st.selectbox("Number of Products", [1,2,3,4])
        has_cc       = st.selectbox("Has Credit Card?", ["Yes","No"])
        is_active    = st.selectbox("Active Member?", ["Yes","No"])

    st.markdown("---")
    predict_btn = st.button("⚡ Calculate Churn Risk", type="primary", use_container_width=True)

    if predict_btn:
        # Build input row matching training features
        input_dict = {
            'CreditScore':             credit_score,
            'Age':                     age,
            'Tenure':                  tenure,
            'Balance':                 balance,
            'NumOfProducts':           num_products,
            'HasCrCard':               1 if has_cc=="Yes" else 0,
            'IsActiveMember':          1 if is_active=="Yes" else 0,
            'EstimatedSalary':         salary,
            'Balance_Salary_Ratio':    balance/(salary+1),
            'Age_Tenure_Interaction':  age*tenure,
            'Product_Active_Interaction': num_products*(1 if is_active=="Yes" else 0),
            'Zero_Balance':            1 if balance==0 else 0,
            'Geography_Germany':       1 if geography=="Germany" else 0,
            'Geography_Spain':         1 if geography=="Spain" else 0,
            'Gender_Male':             1 if gender=="Male" else 0,
        }
        input_df = pd.DataFrame([input_dict])[FEATURES]
        prob     = model.predict_proba(input_df)[0][1]
        pct      = round(prob*100, 1)

        # Risk level
        if prob >= 0.6:
            risk_label = "HIGH RISK"
            risk_color = "#C0392B"
            risk_class = "risk-high"
            risk_icon  = "🔴"
            action     = "Immediate intervention required — assign dedicated relationship manager, offer exclusive loyalty benefits."
        elif prob >= 0.35:
            risk_label = "MEDIUM RISK"
            risk_color = "#D4AC0D"
            risk_class = "risk-medium"
            risk_icon  = "🟡"
            action     = "Proactive engagement recommended — send personalised product offer, schedule check-in call."
        else:
            risk_label = "LOW RISK"
            risk_color = "#1E8449"
            risk_class = "risk-low"
            risk_icon  = "🟢"
            action     = "Customer appears stable — continue standard engagement and monitor quarterly."

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode    = "gauge+number+delta",
            value   = pct,
            number  = {'suffix':'%','font':{'size':48,'color':risk_color}},
            delta   = {'reference':20,'increasing':{'color':'#C0392B'},'decreasing':{'color':'#1E8449'}},
            gauge   = {
                'axis':{'range':[0,100],'tickwidth':1,'tickcolor':'#566573'},
                'bar':{'color':risk_color,'thickness':0.25},
                'bgcolor':'white',
                'borderwidth':2,
                'steps':[
                    {'range':[0,35],   'color':'#EAFAF1'},
                    {'range':[35,60],  'color':'#FEF9E7'},
                    {'range':[60,100], 'color':'#FDEDEC'},
                ],
                'threshold':{'line':{'color':risk_color,'width':4},'thickness':0.75,'value':pct}
            },
            title={'text': f'{risk_icon} CHURN PROBABILITY', 'font':{'size':18,'color':'#1B4F72'}}
        ))
        fig.update_layout(height=320, margin=dict(l=30,r=30,t=50,b=10), paper_bgcolor='white')

        r1, r2 = st.columns([1,1])
        with r1:
            st.plotly_chart(fig, use_container_width=True)
        with r2:
            st.markdown(f"""
            <div class="metric-card {risk_class}" style="margin-top:30px;">
              <div class="val" style="color:{risk_color};">{pct}%</div>
              <div class="lbl">Churn Probability</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="insight-box" style="margin-top:14px;">
              <strong>{risk_icon} {risk_label}</strong><br>{action}
            </div>
            """, unsafe_allow_html=True)

        # Key factors for this customer
        st.markdown('<div class="section-title">Key Risk Factors Detected</div>', unsafe_allow_html=True)
        factors = []
        if age > 45:          factors.append(("🔴 Age above 45", "Older customers show significantly higher churn tendency"))
        if balance > 100000:  factors.append(("🔴 High account balance", "High-balance customers tend to explore better alternatives"))
        if num_products >= 3: factors.append(("🔴 3+ products", "Customers with 3-4 products have very high churn rates"))
        if is_active == "No": factors.append(("🟡 Inactive member", "Inactive customers are more likely to leave"))
        if geography == "Germany": factors.append(("🟡 Germany location", "German customers show higher average churn rates"))
        if balance == 0:      factors.append(("🟡 Zero balance", "Zero-balance accounts signal disengagement"))
        if tenure < 2:        factors.append(("🟡 Low tenure", "New customers have higher churn risk"))
        if not factors:       factors.append(("🟢 No major risk factors", "Customer profile appears stable"))

        for icon_text, desc in factors:
            st.markdown(f"**{icon_text}** — {desc}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Model Performance
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">Model Comparison — All Four Algorithms</div>', unsafe_allow_html=True)

    # Results table
    df_results = pd.DataFrame(results).T.reset_index()
    df_results.columns = ['Model','Accuracy (%)','Precision (%)','Recall (%)','F1-Score (%)','ROC-AUC']
    st.dataframe(df_results.style.highlight_max(
        subset=['Accuracy (%)','Precision (%)','Recall (%)','F1-Score (%)','ROC-AUC'],
        color='#D5F5E3'), use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    <strong>Best Model Selected: Gradient Boosting</strong><br>
    Gradient Boosting achieved the highest ROC-AUC score (0.74), making it the most capable model for
    discriminating between churners and retained customers. ROC-AUC is the primary metric for churn
    prediction because it measures the model's ability to rank at-risk customers correctly — essential for
    prioritising retention campaigns.
    </div>
    """, unsafe_allow_html=True)

    # Visual metric comparison
    st.markdown('<div class="section-title">Visual Performance Comparison</div>', unsafe_allow_html=True)
    models_list = list(results.keys())
    colors = ['#1B4F72','#D4AC0D','#1E8449','#C0392B']

    col_a, col_b = st.columns(2)
    with col_a:
        fig_acc = go.Figure()
        accs = [results[m]['Accuracy'] for m in models_list]
        fig_acc.add_trace(go.Bar(x=models_list, y=accs, marker_color=colors,
                                  text=[f'{v}%' for v in accs], textposition='outside'))
        fig_acc.update_layout(title='Accuracy Comparison', yaxis_range=[50,95],
                               yaxis_title='Accuracy (%)', height=350,
                               plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig_acc, use_container_width=True)

    with col_b:
        fig_auc = go.Figure()
        aucs = [results[m]['ROC-AUC'] for m in models_list]
        fig_auc.add_trace(go.Bar(x=models_list, y=aucs, marker_color=colors,
                                  text=[str(v) for v in aucs], textposition='outside'))
        fig_auc.update_layout(title='ROC-AUC Score Comparison', yaxis_range=[0.5,1.0],
                               yaxis_title='ROC-AUC', height=350,
                               plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig_auc, use_container_width=True)

    # Metric explanation
    st.markdown('<div class="section-title">Understanding the Metrics</div>', unsafe_allow_html=True)
    metric_data = {
        'Metric':      ['Accuracy','Precision','Recall','F1-Score','ROC-AUC'],
        'What it means': [
            'Out of all customers, what % did we classify correctly?',
            'Of customers we predicted as churners, what % actually churned?',
            'Of all actual churners, what % did we correctly identify?',
            'Balance between Precision and Recall — the overall churn detection quality',
            'Model\'s ability to rank churners higher than non-churners — 1.0 is perfect, 0.5 is random'
        ],
        'Why it matters': [
            'General performance measure',
            'Controls false alarms — targeting wrong customers wastes budget',
            'Captures actual churners — missing them costs revenue',
            'Most useful single metric when classes are imbalanced',
            'Best metric for comparing models on imbalanced churn data'
        ]
    }
    st.dataframe(pd.DataFrame(metric_data), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Data Insights
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">Key Findings from Exploratory Data Analysis</div>',
                unsafe_allow_html=True)

    insights = [
        ("🌍 Geography", "German customers churn at significantly higher rates compared to French and Spanish customers. Germany should be a priority geography for retention campaigns."),
        ("👴 Age",       "Customers aged 40–60 show the highest churn tendency. The 50–60 age bracket is particularly high-risk, likely seeking better wealth management alternatives."),
        ("📦 Products",  "Customers with 3 or 4 products have dramatically higher churn than those with 1 or 2. This counter-intuitive finding suggests over-selling products creates dissatisfaction."),
        ("💤 Activity",  "Inactive members churn at nearly double the rate of active members. Member engagement is a strong early-warning signal for churn."),
        ("💰 Balance",   "High-balance customers (above €100,000) show elevated churn risk — they are more financially aware and likely to compare alternatives."),
        ("⏳ Tenure",    "Customers with very low tenure (0–2 years) and those with very high tenure (8+ years) both show elevated churn. Mid-tenure customers are most loyal."),
    ]

    for title, text in insights:
        col_i, col_t = st.columns([1,8])
        with col_t:
            st.markdown(f"**{title}**")
            st.markdown(f'<div class="insight-box">{text}</div>', unsafe_allow_html=True)

    # Feature importance chart (interactive)
    st.markdown('<div class="section-title">Feature Importance — What Drives Churn</div>',
                unsafe_allow_html=True)

    features_imp = {
        'Age':                     0.187,
        'Balance':                 0.142,
        'Age_Tenure_Interaction':  0.118,
        'NumOfProducts':           0.096,
        'Balance_Salary_Ratio':    0.089,
        'CreditScore':             0.076,
        'EstimatedSalary':         0.071,
        'IsActiveMember':          0.068,
        'Tenure':                  0.052,
        'Product_Active_Interaction': 0.038,
        'Geography_Germany':       0.025,
        'Gender_Male':             0.018,
        'HasCrCard':               0.011,
        'Geography_Spain':         0.008,
        'Zero_Balance':            0.006,
    }
    fi_df = pd.DataFrame({'Feature': list(features_imp.keys()),
                           'Importance': list(features_imp.values())}).sort_values('Importance')
    colors_fi = ['#C0392B' if v == max(fi_df['Importance']) else
                  '#D4AC0D' if v >= fi_df['Importance'].quantile(0.75) else '#1B4F72'
                  for v in fi_df['Importance']]
    fig_fi = go.Figure(go.Bar(x=fi_df['Importance'], y=fi_df['Feature'],
                               orientation='h', marker_color=colors_fi,
                               text=[f'{v:.3f}' for v in fi_df['Importance']],
                               textposition='outside'))
    fig_fi.update_layout(height=500, xaxis_title='Importance Score',
                          title='Feature Importance Ranking (Random Forest)',
                          plot_bgcolor='white', paper_bgcolor='white',
                          margin=dict(l=200))
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    <strong>Top 3 Churn Drivers:</strong><br>
    1. <strong>Age</strong> — The single strongest predictor. Older customers (45+) are significantly more likely to leave.<br>
    2. <strong>Account Balance</strong> — High-balance customers have greater financial awareness and more options to explore.<br>
    3. <strong>Age × Tenure Interaction</strong> — A combined signal: older customers who have been with the bank for many years
    may feel the bank no longer meets their evolved financial needs.
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style="text-align:center; color:#7F8C8D; font-size:13px;">
Bank Customer Churn Intelligence System · Built with Gradient Boosting ML ·
Project by Ashwani Singh, B.Com (Hons), Avinash College of Commerce ·
Unified Mentor Internship 2026
</p>
""", unsafe_allow_html=True)
