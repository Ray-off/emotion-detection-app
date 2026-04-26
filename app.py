import streamlit as st
from utils.predict import predict

# Page config
st.set_page_config(
    page_title="Emotion Detection",
    page_icon="💬",
    layout="centered"
)

# Title
st.title("💬 Social Media Emotion Detection")
st.markdown("Detect emotions from text using a fine-tuned BERT model.")

# Input box
user_input = st.text_area("Enter your text here:")

# Button
if st.button("Predict Emotion"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        result = predict(user_input)

        # Show main result
        st.success(f"🎯 Predicted Emotion: **{result['label']}**")
        st.info(f"Confidence: {result['confidence']:.2f}")

        # Show probability distribution
        st.subheader("📊 Emotion Probabilities")
        st.bar_chart(result["all_probs"])