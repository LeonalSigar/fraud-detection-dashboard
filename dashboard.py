import streamlit as st
import pandas as pd

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)

# ========== CUSTOM CSS ==========
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .fraud-alert {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        border-radius: 5px;
    }
    .safe-alert {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ========== HEADER ==========
st.markdown('<p class="main-header">🔍 Real-Time Fraud Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">WQD7008 Parallel & Distributed Computing | Group 14 | GA2</p>', unsafe_allow_html=True)

# ========== ARCHITECTURE INFO ==========
with st.expander("ℹ️ System Architecture", expanded=False):
    st.markdown("""
    **Pipeline Overview:**
```
    Data (S3) → Spark on EMR → ML Model (GBT) → Predictions → Lambda API → Dashboard
```
    
    **Technologies Used:**
    - **AWS S3**: Data storage
    - **AWS EMR**: Spark cluster for distributed processing
    - **PySpark MLlib**: Gradient Boosted Trees model
    - **AWS Lambda**: Prediction API
    - **Streamlit**: Dashboard visualization
    
    **Dataset**: IEEE-CIS Fraud Detection (506,691 transactions)
    """)

st.markdown("---")

# ========== LOAD DATA ==========
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("predictions.csv")
        return df, None
    except FileNotFoundError:
        return None, "predictions.csv not found"
    except Exception as e:
        return None, str(e)

df, error = load_data()

if error:
    st.error(f"Error: {error}")
    st.stop()

# ========== DETECT COLUMNS ==========
df.columns = df.columns.str.strip()

# Find prediction column
pred_col = None
for col in df.columns:
    if 'prediction' in col.lower():
        pred_col = col
        break
if pred_col is None:
    pred_col = df.columns[-1]

# Find ID column
id_col = None
for col in df.columns:
    if 'transaction' in col.lower():
        id_col = col
        break
if id_col is None:
    id_col = df.columns[0]

# ========== CALCULATE METRICS ==========
total = len(df)
fraud = int(df[pred_col].sum())
legit = total - fraud
fraud_rate = (fraud / total) * 100 if total > 0 else 0

# ========== METRICS DISPLAY ==========
st.subheader("📊 Prediction Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Transactions",
        value=f"{total:,}"
    )

with col2:
    st.metric(
        label="✅ Legitimate",
        value=f"{legit:,}",
        delta=f"{100-fraud_rate:.1f}%"
    )

with col3:
    st.metric(
        label="🚨 Fraudulent",
        value=f"{fraud:,}",
        delta=f"{fraud_rate:.1f}%",
        delta_color="inverse"
    )

with col4:
    st.metric(
        label="Model",
        value="GBT",
        delta="AUC-PR: 0.304"
    )

st.markdown("---")

# ========== TRANSACTION LOOKUP ==========
st.subheader("🔎 Transaction Lookup")

col1, col2 = st.columns([2, 3])

with col1:
    search_id = st.text_input(
        "Enter Transaction ID:",
        placeholder="e.g., 3663549",
        help="Enter a transaction ID to check if it's fraudulent"
    )
    
    search_button = st.button("🔍 Check Transaction", type="primary", use_container_width=True)

with col2:
    if search_button and search_id:
        try:
            search_id_int = int(search_id)
            result = df[df[id_col] == search_id_int]
            
            if len(result) > 0:
                prediction = result[pred_col].values[0]
                
                if prediction == 1 or prediction == 1.0:
                    st.markdown("""
                    <div class="fraud-alert">
                        <h3>🚨 FRAUD DETECTED</h3>
                        <p>Transaction <strong>{}</strong> is flagged as <strong>FRAUDULENT</strong></p>
                        <p>Recommended Action: Block transaction and verify with cardholder</p>
                    </div>
                    """.format(search_id), unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="safe-alert">
                        <h3>✅ TRANSACTION SAFE</h3>
                        <p>Transaction <strong>{}</strong> is <strong>LEGITIMATE</strong></p>
                        <p>No action required</p>
                    </div>
                    """.format(search_id), unsafe_allow_html=True)
            else:
                st.warning(f"Transaction ID `{search_id}` not found in dataset")
                
        except ValueError:
            st.error("Please enter a valid numeric Transaction ID")
    
    elif search_button:
        st.warning("Please enter a Transaction ID")

st.markdown("---")

# ========== QUICK TEST IDS ==========
st.subheader("⚡ Quick Test")
st.markdown("Click to test sample transactions:")

col1, col2, col3, col4 = st.columns(4)

# Find sample IDs
legit_ids = df[df[pred_col] == 0][id_col].head(2).tolist()
fraud_ids = df[df[pred_col] == 1][id_col].head(2).tolist()

with col1:
    if len(legit_ids) > 0:
        if st.button(f"✅ {legit_ids[0]}", use_container_width=True):
            st.success(f"Transaction {legit_ids[0]}: LEGITIMATE")

with col2:
    if len(legit_ids) > 1:
        if st.button(f"✅ {legit_ids[1]}", use_container_width=True):
            st.success(f"Transaction {legit_ids[1]}: LEGITIMATE")

with col3:
    if len(fraud_ids) > 0:
        if st.button(f"🚨 {fraud_ids[0]}", use_container_width=True):
            st.error(f"Transaction {fraud_ids[0]}: FRAUDULENT")

with col4:
    if len(fraud_ids) > 1:
        if st.button(f"🚨 {fraud_ids[1]}", use_container_width=True):
            st.error(f"Transaction {fraud_ids[1]}: FRAUDULENT")

st.markdown("---")

# ========== VISUALIZATION ==========
st.subheader("📈 Prediction Distribution")

col1, col2 = st.columns(2)

with col1:
    chart_data = pd.DataFrame({
        'Status': ['Legitimate', 'Fraudulent'],
        'Count': [legit, fraud]
    })
    st.bar_chart(chart_data.set_index('Status'), color=["#4CAF50"])

with col2:
    st.markdown("**Statistics:**")
    st.markdown(f"""
    | Metric | Value |
    |--------|-------|
    | Total Processed | {total:,} |
    | Legitimate | {legit:,} ({100-fraud_rate:.2f}%) |
    | Fraudulent | {fraud:,} ({fraud_rate:.2f}%) |
    | Model Used | Gradient Boosted Trees |
    | Processing | AWS EMR (Spark) |
    """)

st.markdown("---")

# ========== SAMPLE DATA ==========
st.subheader("📋 Sample Predictions")

tab1, tab2 = st.tabs(["🚨 Fraudulent Transactions", "✅ Legitimate Transactions"])

with tab1:
    fraud_samples = df[df[pred_col] == 1].head(10)
    if len(fraud_samples) > 0:
        st.dataframe(fraud_samples, use_container_width=True, hide_index=True)
    else:
        st.info("No fraudulent transactions found")

with tab2:
    legit_samples = df[df[pred_col] == 0].head(10)
    st.dataframe(legit_samples, use_container_width=True, hide_index=True)

# ========== FOOTER ==========
st.markdown("""
<div style='text-align: center; color: #888; padding: 1rem;'>
    <p><strong>WQD7008 Parallel & Distributed Computing</strong></p>
    <p>Group 14: Leo | Sugin | Nirvar | Luqman | Shi Hui</p>
    <p>Universiti Malaya | January 2026</p>
</div>
""", unsafe_allow_html=True)

---

## REQUIREMENTS.TXT
```
streamlit==1.31.0
pandas==2.0.3
