import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# NLTK downloads
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Sidebar menu
with st.sidebar:
    selection = option_menu(
        "Choose The Prediction",
        ['Fraud Claim Detection', 'Customer Segmenation', 'Sentiment Analysis',
         'Insurance Risk_Score', 'Insurance Claim_Amount', 'Text Translator'],
        menu_icon='cash-stack',
        icons=['person-fill', 'people-fill', 'graph-up', 'cash', 'currency-dollar', 'globe'],
        default_index=0
    )
# Set the background image
def set_bg_image(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Example usage
set_bg_image("https://www.humanitas.net/content/uploads/2023/11/medicine-5.jpg")

# Load Models
with open('frauddetection.pkl', 'rb') as file:
    model_fraud = pickle.load(file)
with open('customersegmentation.pkl', 'rb') as file:
    model_segment = pickle.load(file)
with open('riskClassification1.pkl', 'rb') as file:
    model_risk = pickle.load(file)
with open("scalerrisk1.pkl", "rb") as scaler_file:
    scaler_risk = pickle.load(scaler_file)
with open("ClaimRegression1.pkl", 'rb') as file:
    model_claim = pickle.load(file)
with open("scalerclaim1.pkl", "rb") as scaler_file:
    scaler_claim = pickle.load(scaler_file)

# Fraud Claim Detection
if selection == "Fraud Claim Detection":
    st.header("🛡️Fraud Claim Detection")
    
    col1, col2 = st.columns(2)
    with col1:
        Claim_Amount = st.number_input("**Claim Amount**", min_value=0.0, step=100.0)
        Suspicious_Flags = st.selectbox("**Suspicious Flags**", [0, 1])
        Claim_Type_Home = st.selectbox("**Claim Type Home**", [0, 1])
    with col2:
        Claim_Type_Medical = st.selectbox("**Claim Type Medical**", [0, 1])
        Claim_Type_Vehicle = st.selectbox("**Claim Type Vehicle**", [0, 1])

    input_data = np.array([[Claim_Amount, Suspicious_Flags, Claim_Type_Home,
                            Claim_Type_Medical, Claim_Type_Vehicle]])

    if st.button("**🧠 Predict**"):
        prediction = model_fraud.predict(input_data)
        if prediction[0] == 1:
            st.error("🚨 Prediction: Fraudulent Claim Detected!")
        else:
            st.success("✅ Prediction: Genuine Claim")

# Customer Segmentation
elif selection == 'Customer Segmenation':
    st.header("👥Customer Segmenation")
   
    Policy_Count = st.number_input("**Policy_Count**")
    Claim_Frequency = st.number_input("**Claim_Frequency**")
    Policy_Upgrades = st.number_input("**Policy_Upgrades**")
    Kmeans_Cluster = st.number_input("**Kmeans_Cluster**")

    inputdata1 = np.array([[Policy_Count, Claim_Frequency, Policy_Upgrades, Kmeans_Cluster]])
    feature_names = model_segment.feature_names_in_
    inputdata_df1 = pd.DataFrame(inputdata1, columns=feature_names)

    if st.button("**🧠 Predict**"):
        prediction = model_segment.predict(inputdata_df1)
        st.success(f"Customer Segment: {prediction[0]}")

# Sentiment Analysis
elif selection == 'Sentiment Analysis':
    st.header("😃😐😠Sentiment Analysis")
    
    df = pd.read_csv("insurance_reviews.csv")

    def assign(Rating):
        if Rating in [1, 2]:
            return "Negative"
        elif Rating == 3:
            return "Neutral"
        else:
            return "Positive"

    df['Sentimentlabel'] = df['Rating'].apply(assign)
    df = df[["Review_Text", "Sentimentlabel"]]

    stop_words = set(stopwords.words('english'))
    lemma = WordNetLemmatizer()

    def clean(text):
        text = re.sub(r'https\S+', '', text)
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = text.lower()
        text = word_tokenize(text)
        text = [lemma.lemmatize(word, pos='v') for word in text if word not in stop_words and len(word) > 2]
        return ' '.join(text)

    df['customerfeedback'] = df['Review_Text'].apply(clean)
    df = df.drop('Review_Text', axis=1)

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['customerfeedback'])
    y = df['Sentimentlabel']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_sentiment = MultinomialNB()
    model_sentiment.fit(X_train, y_train)
    y_pred = model_sentiment.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    def predict_sentiment(text):
        processed = clean(text)
        vectorized = vectorizer.transform([processed])
        return model_sentiment.predict(vectorized)[0]

    feedback = st.text_input("**Enter your feedback:**")
    if st.button("**Submit**"):
        sentiment = predict_sentiment(feedback)
        st.write(f"Accuracy: {accuracy * 100:.2f}%")
        st.write(f"Predicted Sentiment: **{sentiment}**")

    wc = WordCloud(width=800, height=300, background_color='white').generate(' '.join(df['customerfeedback']))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

# Insurance Risk Score
elif selection == 'Insurance Risk_Score':
    st.header("⚠️📉Insurance Risk_Score")
   
    col1, col2 = st.columns(2)
    with col1:
        Age = st.number_input("**Age**", key="age_risk")
        Annual_Income = st.number_input("**Annual Income**", key="income_risk")
        Claim_History = st.number_input("**Claim History**", key="ch_risk")
        Claim_Amount = st.number_input("**Claim Amount**", key="ca_risk")
        Fraud_Label = st.number_input("**Fraud Label**", key="fl_risk")
        Premium_Amount = st.number_input("**Premium Amount**", key="pa_risk")
    with col2:
        Policy_Type_Auto = st.number_input("**Policy Type: Auto**", key="pt_auto_risk")
        Policy_Type_Health = st.number_input("**Policy Type: Health**", key="pt_health_risk")
        Policy_Type_Life = st.number_input("**Policy Type: Life**", key="pt_life_risk")
        Policy_Type_Property = st.number_input("**Policy Type: Property**", key="pt_prop_risk")
        Gender_Female = st.number_input("**Gender: Female**", key="gf_risk")
        Gender_Male = st.number_input("**Gender: Male**", key="gm_risk")

    inputdata3 = np.array([[Age, Annual_Income, Claim_History, Claim_Amount, Fraud_Label, Premium_Amount,
                            Policy_Type_Auto, Policy_Type_Health, Policy_Type_Life, Policy_Type_Property,
                            Gender_Female, Gender_Male]])
    

    try:
        input_scaled = scaler_risk.transform(inputdata3)
        st.write("✅ Scaled input data:", input_scaled)

        if st.button("**🧠 Predict**", key="risk_button"):
            prediction = model_risk.predict(input_scaled)[0]
            st.write(prediction)

            if prediction == "Low":
                st.success("🟢 Risk Score: Low")
            elif prediction == "Medium":
                st.warning("🟡 Risk Score: Medium")
            else:
                st.error("🔴 Risk Score: High")

    except Exception as e:
        st.error(f"🚫 Error during prediction: {e}")

# Insurance Claim Amount
elif selection == 'Insurance Claim_Amount':
    st.header("💸Insurance Claim_Amount")
    
    col1, col2 = st.columns(2)
    with col1:
        Age = st.number_input("**Age**", key="age_claim")
        Annual_Income = st.number_input("**Annual Income**", key="income_claim")
        Claim_History = st.number_input("**Claim History**", key="ch_claim")
        Risk_Score = st.number_input("**Risk Score**", key="rs_claim")
        Fraud_Label = st.number_input("**Fraud Label**", key="fl_claim")
        Premium_Amount = st.number_input("**Premium Amount**", key="pa_claim")
    with col2:
        Policy_Type_Auto = st.number_input("**Policy Type: Auto**", key="pt_auto_claim")
        Policy_Type_Health = st.number_input("**Policy Type: Health**", key="pt_health_claim")
        Policy_Type_Life = st.number_input("**Policy Type: Life**", key="pt_life_claim")
        Policy_Type_Property = st.number_input("**Policy Type: Property**", key="pt_prop_claim")
        Gender_Female = st.number_input("**Gender: Female**", key="gf_claim")
        Gender_Male = st.number_input("**Gender: Male**", key="gm_claim")

    inputdata5 = np.array([[Age, Annual_Income, Claim_History, Risk_Score, Fraud_Label, Premium_Amount,
                            Policy_Type_Auto, Policy_Type_Health, Policy_Type_Life, Policy_Type_Property,
                            Gender_Female, Gender_Male]])
    input_scaled1 = scaler_claim.transform(inputdata5)
    
    if st.button("**🧠 Predict**", key="claim_button"):
        prediction = model_claim.predict(input_scaled1)[0]
        st.write(prediction)
        st.success(f"💰 Predicted Claim Amount: ₹{prediction:,.2f}")

# Text Translator
elif selection == "Text Translator":
    st.header("\U0001F30D Insurance Text Translator")

    @st.cache_resource
    def load_translation_model():
        model_name = "facebook/m2m100_418M"
        tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        return tokenizer, model

    tokenizer, model = load_translation_model()

    def translate_text(text, src_lang, tgt_lang):
        tokenizer.src_lang = src_lang
        encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.get_lang_id(tgt_lang)
        )
        return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    uploaded_file = st.file_uploader("Upload cleaned_insurance_text.csv", type=["csv"])
    if uploaded_file:
        df1 = pd.read_csv(uploaded_file)
        st.write("\U0001F4C4 Preview of Uploaded File:", df1.head())

        if st.button("🔁 Translate to Tamil & French"):
            with st.spinner("Translating... Please wait ⏳"):
                df1["cleanedtext_ta"] = df1["cleanedtext"].apply(lambda x: translate_text(str(x), "en", "ta"))
                df1["cleanedtext_fr"] = df1["cleanedtext"].apply(lambda x: translate_text(str(x), "en", "fr"))
                df1["cleanedsummary_ta"] = df1["cleanedsummary"].apply(lambda x: translate_text(str(x), "en", "ta"))
                df1["cleanedsummary_fr"] = df1["cleanedsummary"].apply(lambda x: translate_text(str(x), "en", "fr"))

            st.success("✅ Translation Complete!")
            st.dataframe(df1.head())

            csv = df1.to_csv(index=False, encoding='utf-8')
            st.download_button(
                label="\U0001F4C5 Download Translated File",
                data=csv,
                file_name="translated_insurance_text.csv",
                mime="text/csv"
            )
