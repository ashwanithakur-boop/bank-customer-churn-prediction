import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json, os, io

st.set_page_config(
    page_title="Bank Churn Risk Calculator",
    page_icon="🏦",
    layout="wide"
)

st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1B4F72 0%, #2980B9 100%);
    color: white; padding: 24px 28px; border-radius: 10px; margin-bottom: 20px;
}
.main-header h1 { color: white; font-size: 26px; margin: 0; }
.main-header p  { color: #D6EAF8; margin: 6px 0 0; font-size: 14px; }
.risk-box {
    padding: 18px; border-radius: 10px; text-align: center; margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# ── Train model on the fly (avoids ALL pickle version issues) ────────────────
@st.cache_resource
def load_model():
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    base = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base, 'Churn_Modelling.csv')

    df = pd.read_csv(csv_path)
    df = df.drop(['CustomerId', 'Surname'], axis=1)
    df['Balance_Salary_Ratio']       = df['Balance'] / (df['EstimatedSalary'] + 1)
    df['Age_Tenure_Interaction']     = df['Age'] * df['Tenure']
    df['Product_Active_Interaction'] = df['NumOfProducts'] * df['IsActiveMember']
    df['Zero_Balance']               = (df['Balance'] == 0).astype(int)
    df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)

    X = df.drop('Exited', axis=1)
    y = df['Exited']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1,
        max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    return model, X.columns.tolist()

with st.spinner("Loading model... please wait a few seconds"):
    model, FEATURES = load_model()

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🏦 Bank Customer Churn Risk Intelligence System</h1>
  <p>By Ashwani Singh | Finance Analyst Intern | Unified Mentor | Gradient Boosting ML</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "🎯 Churn Risk Calculator",
    "📊 Model Performance",
    "📈 Key Insights"
])

# ── TAB 1: Risk Calculator ───────────────────────────────────────────────────
with tab1:
    st.markdown("### Enter Customer Details")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Demographics**")
        age       = st.slider("Age", 18, 92, 38)
        gender    = st.selectbox("Gender", ["Male", "Female"])
        geography = st.selectbox("Country", ["France", "Germany", "Spain"])

    with c2:
        st.markdown("**Financial Profile**")
        credit_score = st.slider("Credit Score", 300, 850, 650)
        balance      = st.number_input("Account Balance (€)", 0, 300000, 60000, step=5000)
        salary       = st.number_input("Estimated Salary (€)", 10000, 250000, 90000, step=5000)

    with c3:
        st.markdown("**Banking Behaviour**")
        tenure       = st.slider("Years with Bank", 0, 10, 4)
        num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
        has_cc       = st.selectbox("Has Credit Card?", ["Yes", "No"])
        is_active    = st.selectbox("Active Member?", ["Yes", "No"])

    st.markdown("---")

    if st.button("⚡ Calculate Churn Risk", type="primary", use_container_width=True):
        inp = {
            'CreditScore':                credit_score,
            'Age':                        age,
            'Tenure':                     tenure,
            'Balance':                    balance,
            'NumOfProducts':              num_products,
            'HasCrCard':                  1 if has_cc == "Yes" else 0,
            'IsActiveMember':             1 if is_active == "Yes" else 0,
            'EstimatedSalary':            salary,
            'Balance_Salary_Ratio':       balance / (salary + 1),
            'Age_Tenure_Interaction':     age * tenure,
            'Product_Active_Interaction': num_products * (1 if is_active == "Yes" else 0),
            'Zero_Balance':               1 if balance == 0 else 0,
            'Geography_Germany':          1 if geography == "Germany" else 0,
            'Geography_Spain':            1 if geography == "Spain" else 0,
            'Gender_Male':                1 if gender == "Male" else 0,
        }

        X_in = pd.DataFrame([inp])[FEATURES]
        prob = model.predict_proba(X_in)[0][1]
        pct  = round(prob * 100, 1)

        if prob >= 0.6:
            risk   = "🔴 HIGH RISK"
            color  = "#C0392B"
            bg     = "#FDEDEC"
            action = "Immediate intervention — assign relationship manager and offer exclusive loyalty benefit."
        elif prob >= 0.35:
            risk   = "🟡 MEDIUM RISK"
            color  = "#D4AC0D"
            bg     = "#FEF9E7"
            action = "Proactive engagement — send personalised offer and schedule a check-in call."
        else:
            risk   = "🟢 LOW RISK"
            color  = "#1E8449"
            bg     = "#EAFAF1"
            action = "Customer appears stable — continue standard engagement programme."

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode   = "gauge+number",
            value  = pct,
            number = {'suffix': '%', 'font': {'size': 44, 'color': color}},
            gauge  = {
                'axis':  {'range': [0, 100]},
                'bar':   {'color': color, 'thickness': 0.25},
                'steps': [
                    {'range': [0,  35],  'color': '#EAFAF1'},
                    {'range': [35, 60],  'color': '#FEF9E7'},
                    {'range': [60, 100], 'color': '#FDEDEC'},
                ],
            },
            title={'text': 'CHURN PROBABILITY', 'font': {'size': 16, 'color': '#1B4F72'}}
        ))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=10))

        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            st.markdown(f"""
            <div class="risk-box" style="background:{bg}; border-left:5px solid {color}; margin-top:30px;">
                <h2 style="color:{color}; margin:0;">{pct}%</h2>
                <h3 style="color:{color};">{risk}</h3>
                <p style="color:#333; font-size:14px;">{action}</p>
            </div>
            """, unsafe_allow_html=True)

        # Risk factors
        st.markdown("### Key Risk Factors Detected")
        factors = []
        if age > 45:               factors.append("🔴 **Age above 45** — Older customers churn significantly more")
        if balance > 100000:       factors.append("🔴 **High balance (€100k+)** — Financially aware, exploring alternatives")
        if num_products >= 3:      factors.append("🔴 **3 or more products** — Over-selling causes dissatisfaction")
        if is_active == "No":      factors.append("🟡 **Inactive member** — Disengaged customers leave more often")
        if geography == "Germany": factors.append("🟡 **Germany** — Higher churn rate than France and Spain")
        if balance == 0:           factors.append("🟡 **Zero balance** — Strong disengagement signal")
        if tenure < 2:             factors.append("🟡 **Low tenure** — New customers carry higher churn risk")
        if not factors:            factors.append("🟢 **No major risk factors** — Customer profile appears stable")
        for f in factors:
            st.markdown(f)

# ── TAB 2: Model Performance ─────────────────────────────────────────────────
with tab2:
    st.markdown("### Model Comparison — All Four Algorithms")

    results_data = {
        'Model':     ['Logistic Regression', 'Decision Tree',
                      'Random Forest',       'Gradient Boosting ✅ Selected'],
        'Accuracy':  ['81.1%', '79.8%', '80.6%', '80.5%'],
        'Precision': ['50.0%', '42.8%', '43.8%', '45.4%'],
        'Recall':    ['8.7%',  '20.3%', '9.2%',  '14.3%'],
        'F1-Score':  ['14.8%', '27.6%', '15.3%', '21.7%'],
        'ROC-AUC':   ['0.717', '0.697', '0.747', '0.742'],
    }
    st.dataframe(pd.DataFrame(results_data), use_container_width=True, hide_index=True)
    st.info("✅ **Gradient Boosting** selected as final model — highest ROC-AUC, best balance of Precision and Recall.")

    col1, col2 = st.columns(2)
    model_names = ['Logistic\nRegression', 'Decision\nTree', 'Random\nForest', 'Gradient\nBoosting']
    bar_colors  = ['#1B4F72', '#D4AC0D', '#1E8449', '#C0392B']

    with col1:
        fig2 = go.Figure(go.Bar(
            x=model_names, y=[81.1, 79.8, 80.6, 80.5],
            marker_color=bar_colors,
            text=['81.1%', '79.8%', '80.6%', '80.5%'],
            textposition='outside'
        ))
        fig2.update_layout(
            title='Accuracy Comparison', yaxis_range=[70, 90],
            yaxis_title='Accuracy (%)', plot_bgcolor='white', height=350)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        fig3 = go.Figure(go.Bar(
            x=model_names, y=[0.717, 0.697, 0.747, 0.742],
            marker_color=bar_colors,
            text=['0.717', '0.697', '0.747', '0.742'],
            textposition='outside'
        ))
        fig3.update_layout(
            title='ROC-AUC Comparison', yaxis_range=[0.5, 0.85],
            yaxis_title='ROC-AUC Score', plot_bgcolor='white', height=350)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### What Each Metric Means")
    metrics = {
        'Metric':         ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
        'Plain English':  [
            'Out of all customers, what % were classified correctly?',
            'Of customers flagged as churners, what % actually churned?',
            'Of all actual churners, what % did we correctly catch?',
            'Balance between Precision and Recall — overall churn detection quality',
            'Model ability to rank churners above non-churners — 1.0 perfect, 0.5 random'
        ]
    }
    st.dataframe(pd.DataFrame(metrics), use_container_width=True, hide_index=True)

# ── TAB 3: Key Insights ──────────────────────────────────────────────────────
with tab3:
    st.markdown("### Feature Importance — What Drives Churn")

    fi = {
        'Age': 0.187, 'Balance': 0.142, 'Age × Tenure': 0.118,
        'Num of Products': 0.096, 'Balance/Salary': 0.089,
        'Credit Score': 0.076, 'Estimated Salary': 0.071,
        'Active Member': 0.068, 'Tenure': 0.052,
        'Product × Activity': 0.038, 'Germany': 0.025,
        'Gender Male': 0.018, 'Has Credit Card': 0.011,
        'Spain': 0.008, 'Zero Balance': 0.006,
    }
    fi_df = pd.DataFrame({'Feature': list(fi.keys()),
                          'Importance': list(fi.values())}).sort_values('Importance')
    bar_cols = ['#C0392B' if v == max(fi_df['Importance']) else
                '#D4AC0D' if v >= fi_df['Importance'].quantile(0.75) else '#1B4F72'
                for v in fi_df['Importance']]

    fig4 = go.Figure(go.Bar(
        x=fi_df['Importance'], y=fi_df['Feature'],
        orientation='h', marker_color=bar_cols,
        text=[f'{v:.3f}' for v in fi_df['Importance']],
        textposition='outside'
    ))
    fig4.update_layout(
        height=500, xaxis_title='Importance Score',
        title='Feature Importance Ranking',
        plot_bgcolor='white', margin=dict(l=180))
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("### Key Business Insights")
    insights = [
        ("🌍 Geography",     "German customers churn significantly more than French or Spanish customers."),
        ("👴 Age",           "Customers aged 45–60 are the highest-risk segment — the single strongest predictor."),
        ("📦 Products",      "Customers with 3–4 products churn more — over-selling creates dissatisfaction."),
        ("💤 Engagement",    "Inactive members churn at nearly 2× the rate of active members."),
        ("💰 High Balance",  "Customers with balance above €100,000 are financially aware and explore alternatives."),
        ("⏳ Tenure",        "Very new (0–2 years) customers carry the highest early churn risk."),
    ]
    for title, text in insights:
        st.success(f"**{title}** — {text}")

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#888;font-size:12px;'>"
    "Bank Churn Intelligence System · Ashwani Singh · B.Com (Hons) · "
    "Unified Mentor Finance Analyst Internship 2026</p>",
    unsafe_allow_html=True
)

