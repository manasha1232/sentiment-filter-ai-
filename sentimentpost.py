import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# --- Load Models ---
analyzer = SentimentIntensityAnalyzer()
rewriter = pipeline("text2text-generation", model="t5-small")  # rewrite engine

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="Sentiment Filter AI",
    page_icon="🧠",
    layout="centered"
)

st.title("😎 Sentiment Filter AI")
st.write("Create posts that stay **positive**. If negativity > 50%, we help you rewrite it.")

# --- Input Box ---
user_text = st.text_area("✏️ Write your post:", height=200)

if st.button("Analyze Sentiment"):
    if not user_text.strip():
        st.warning("Please enter text!")
    else:
        scores = analyzer.polarity_scores(user_text)
        neg = round(scores["neg"] * 100, 2)
        neu = round(scores["neu"] * 100, 2)
        pos = round(scores["pos"] * 100, 2)

        # --- Show results ---
        st.subheader("📊 Sentiment Results")
        st.write(f"**Negative:** {neg}%")
        st.progress(min(int(neg), 100))

        st.write(f"**Neutral:** {neu}%")
        st.progress(min(int(neu), 100))

        st.write(f"**Positive:** {pos}%")
        st.progress(min(int(pos), 100))

        # --- If NEG > 50 ---
        if neg >= 50:
            st.error("⚠️ Your post is too negative. Want to rewrite it?")
            
            if st.button("Yes, Rewrite"):
                st.subheader("✨ Rewrite Options")

                # VERSION 1: Soft Positive
                v1 = rewriter(
                    f"Rewrite this to be polite but slightly positive: {user_text}",
                    max_length=120
                )[0]["generated_text"]

                # VERSION 2: Neutral + Clear
                v2 = rewriter(
                    f"Rewrite this in a neutral professional tone: {user_text}",
                    max_length=120
                )[0]["generated_text"]

                # VERSION 3: Fully Positive
                v3 = rewriter(
                    f"Rewrite this to be motivating and positive: {user_text}",
                    max_length=120
                )[0]["generated_text"]

                st.write("#### 🌿 Soft Positive Version")
                st.success(v1)

                st.write("#### 📘 Neutral Professional Version")
                st.info(v2)

                st.write("#### 🌟 Fully Positive Version")
                st.success(v3)
        else:
            st.success("🎉 Your post is fine to publish!")
