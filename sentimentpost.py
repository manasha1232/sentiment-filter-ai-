import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ---------------------------
# LOAD MODELS
# ---------------------------
analyzer = SentimentIntensityAnalyzer()

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
rewrite_model = AutoModelForCausalLM.from_pretrained("distilgpt2")


# ---------------------------
# SENTIMENT FUNCTION
# ---------------------------
def get_sentiment_scores(text):
    score = analyzer.polarity_scores(text)
    neg = round(score["neg"] * 100, 1)
    neu = round(score["neu"] * 100, 1)
    pos = round(score["pos"] * 100, 1)
    return neg, neu, pos


# ---------------------------
# REWRITE FUNCTION
# ---------------------------
def rewrite_positive(text, diversity=0.8):
    prompt = f"Rewrite this in a more positive, calm tone:\n{text}\nRewritten:"

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = rewrite_model.generate(
        **inputs,
        max_length=150,
        do_sample=True,
        temperature=diversity,
        top_p=0.9
    )

    rewritten = tokenizer.decode(outputs[0], skip_special_tokens=True)
    rewritten = rewritten.split("Rewritten:")[-1].strip()
    return rewritten


# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("🔍 Social Media Sentiment Checker + AI Rewriter")

user_text = st.text_area("Enter your post here:", height=150)

if st.button("Analyze Sentiment"):
    if user_text.strip() == "":
        st.warning("Please type something.")
    else:
        neg, neu, pos = get_sentiment_scores(user_text)

        st.subheader("📊 Sentiment Breakdown")
        st.write(f"**Negative:** {neg}%")
        st.write(f"**Neutral:** {neu}%")
        st.write(f"**Positive:** {pos}%")

        if neg > 50:
            st.error("⚠️ Your post has **more than 50% negativity**.")
            rewrite_choice = st.radio(
                "Do you want to rewrite this in a positive tone?",
                ["No", "Yes"]
            )

            if rewrite_choice == "Yes":
                st.subheader("✨ Rewritten Options")

                for i in range(3):  
                    # generate 3 different rewrites
                    output = rewrite_positive(user_text, diversity=0.7 + i*0.1)
                    st.write(f"**Option {i+1}:** {output}")
                    st.markdown("---")

        else:
            st.success("✅ Negativity is within safe limits. No rewrite needed.")
