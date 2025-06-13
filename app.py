import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Wazuh Threat Detector", layout="centered")

st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    h1 {
        color: #004aad;
        text-align: center;
    }
    .stButton>button {
        background-color: #004aad;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚨 Wazuh Threat Detection System")
st.subheader("📊 Upload Wazuh Logs & Predict Threats")

# Step 1: Upload CSV File
uploaded_file = st.file_uploader("📁 Upload Wazuh CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.success("✅ File successfully loaded.")
    st.dataframe(df.head(5))

    # Rename and clean
    df = df.rename(columns={
        '_source.data.url': 'url',
        '_source.rule.firedtimes': 'firedtimes',
        '_source.rule.level': 'level',
        'label': 'label'
    })

    df['label'] = df['label'].map({'real threat': 1, 'false positive': 0})
    df.dropna(subset=['url', 'firedtimes', 'level', 'label'], inplace=True)

    # Data split
    X = df[['url', 'firedtimes', 'level']]
    y = df['label']

    # Preprocessing and pipeline
    preprocessor = ColumnTransformer([
        ('text', TfidfVectorizer(ngram_range=(1, 3), token_pattern=r"(?u)\b\w+\b"), 'url'),
        ('num', StandardScaler(), ['firedtimes', 'level'])
    ])

    pipeline = make_pipeline(preprocessor, DecisionTreeClassifier(random_state=42))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    pipeline.fit(X_train, y_train)

    # Evaluation
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    st.subheader("📈 Model Performance Metrics")
    st.dataframe(pd.DataFrame(report).transpose())

    # Plot confusion matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['False Pos', 'Real Threat'], yticklabels=['False Pos', 'Real Threat'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    # Sample Prediction Input
    st.subheader("🔐 Check New Log Entry")

    with st.form("prediction_form"):
        url_input = st.text_input("Log URL (e.g. /search?q=' OR 1=1;--)", value="/search?q=' OR 1=1;--")
        firedtimes_input = st.number_input("Fired Times", min_value=0, max_value=100, value=5)
        level_input = st.number_input("Threat Level", min_value=0, max_value=20, value=10)
        submitted = st.form_submit_button("Predict")

    if submitted:
        sample = pd.DataFrame([{
            'url': url_input,
            'firedtimes': firedtimes_input,
            'level': level_input
        }])

        prediction = pipeline.predict(sample)
        label = "✅ Real Threat" if prediction[0] == 1 else "🛡️ False Positive"

        st.markdown(f"<h3 style='color:#004aad;'>Prediction Result: {label}</h3>", unsafe_allow_html=True)
