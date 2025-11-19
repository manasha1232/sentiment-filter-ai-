import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.title("Negativity Detection & Rewrite App")

analyzer = SentimentIntensityAnalyzer()

text = st.text_area("Enter text to analyze:")

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        scores = analyzer.polarity_scores(text)
        neg_score = scores['neg'] * 100  # convert to %

        st.subheader("Sentiment Scores")
        st.write(f"Negativity: **{neg_score:.2f}%**")

        # BLOCK IF NEGATIVITY > 60%
        if neg_score > 60:
            st.error("❌ Text blocked because negativity is too high (> 60%).")
            st.stop()

        # Otherwise allow rewrite
        st.success("Negativity is below limit. You can rewrite this text.")

        if st.button("Rewrite to Reduce Negativity"):
            # SIMPLE AI REWRITE (rule-based, no transformers)
            # Converts negative words into softer/positive ones
            replacements = {
                "hate": "dislike",
                "stupid": "not very helpful",
                "angry": "upset",
                "bad": "not ideal",
                "worst": "least preferred",
                "terrible": "difficult",
                "awful": "unpleasant",
                "disgusting": "unpleasant",
                "idiot": "person",
                "dumb": "uninformed",
                "useless": "not effective"
            }

            rewritten = text
            for word, new_word in replacements.items():
                rewritten = rewritten.replace(word, new_word)
                rewritten = rewritten.replace(word.capitalize(), new_word.capitalize())

            st.subheader("Rewritten Text (Less Negative)")
            st.write(rewritten)
