import streamlit as st
from utils import extract_text_from_pdf
from compare_agent import image_to_base64, get_gpt_vision_comparison

st.set_page_config(page_title="Equipment Compliance Checker", layout="centered")

st.title("🧠 Equipment vs. Submittal Compliance (GPT-4o Vision)")

api_key = st.text_input("🔐 Enter your OpenAI API Key", type="password")

uploaded_pdf = st.file_uploader("📄 Upload Technical Submittal PDF", type="pdf")
uploaded_image = st.file_uploader("🖼️ Upload Equipment Nameplate Image", type=["jpg", "jpeg", "png"])

if st.button("Run Comparison") and uploaded_pdf and uploaded_image and api_key:
    with st.spinner("🔍 Analyzing..."):
        submittal_text = extract_text_from_pdf(uploaded_pdf)
        image_b64 = image_to_base64(uploaded_image)
        result_table = get_gpt_vision_comparison(api_key, submittal_text, image_b64)
        st.markdown("### ✅ Compliance Results:")
        st.markdown(result_table)
