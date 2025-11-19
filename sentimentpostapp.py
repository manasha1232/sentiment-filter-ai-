import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

st.set_page_config(page_title="Negativity Gate + Rewriter", layout="centered")
st.title("Negativity Gate 🔒 + Rewriter ✍️")
st.caption("If negativity > 50% the post is considered too negative. You can request rewrites.")

analyzer = SentimentIntensityAnalyzer()

# --- helper functions ---
def clean_text(t: str) -> str:
    t = t.strip()
    # basic cleanup (keeps punctuation for readability)
    t = re.sub(r"\s+", " ", t)
    return t

def neg_percentage(text: str) -> float:
    scores = analyzer.polarity_scores(text)
    return scores["neg"] * 100, scores  # return neg% and full scores

# Replacement helper that keeps case
def replace_word_preserve_case(text: str, old: str, new: str) -> str:
    def repl(match):
        w = match.group(0)
        if w.isupper():
            return new.upper()
        if w[0].isupper():
            return new.capitalize()
        return new
    pattern = r"\b" + re.escape(old) + r"\b"
    return re.sub(pattern, repl, text, flags=re.IGNORECASE)

# Generate three rewrite styles
def generate_rewrites(original: str):
    orig = original.strip()

    # Mapping sets for different rewrite styles
    soft_map = {
        "hate": "really dislike",
        "stupid": "unhelpful",
        "idiot": "person",
        "f**k": "damn",  # example; you can expand
        "fuck": "damn",
        "shit": "mess",
        "awful": "poor",
        "terrible": "very poor",
        "useless": "not very useful",
        "worst": "very disappointing",
        "sucks": "is disappointing",
        "damn": "frustrating"
    }

    neutral_map = {
        "hate": "strongly dislike",
        "stupid": "not suitable",
        "idiot": "individual",
        "fuck": "very frustrating",
        "shit": "problematic",
        "awful": "unsatisfactory",
        "terrible": "substandard",
        "useless": "ineffective",
        "worst": "below expectations",
        "sucks": "is problematic",
        "damn": "quite upsetting"
    }

    positive_map = {
        "hate": "am frustrated with",
        "stupid": "could be improved",
        "idiot": "someone who was mistaken",
        "fuck": "very frustrating",
        "shit": "an issue",
        "awful": "needs improvement",
        "terrible": "could be better",
        "useless": "not helpful in its current form",
        "worst": "not ideal",
        "sucks": "is disappointing",
        "damn": "quite frustrating"
    }

    # 1) Soft rewrite: just replace harsh words and add a softening phrase
    s = orig
    for k, v in soft_map.items():
        s = replace_word_preserve_case(s, k, v)
    # add a softener if sentence starts with a harsh phrase
    if re.search(r"^(i|I'm|i am)\s+(so|very)?\s*(angry|furious|furious|upset|pissed)", s, re.IGNORECASE):
        s = re.sub(r"^(i|I'm|i am)\s+", r"I am ", s, flags=re.IGNORECASE)
    soft_version = s
    if soft_version == orig:
        soft_version = "I feel upset about this and would prefer if it were improved."

    # 2) Neutral rewrite: factual tone, remove insults, suggest improvement
    n = orig
    for k, v in neutral_map.items():
        n = replace_word_preserve_case(n, k, v)
    # make it more neutral: replace "you/they are" -> "there were"
    n = re.sub(r"\b(you|they)\s+are\b", "there were", n, flags=re.IGNORECASE)
    neutral_version = n
    if neutral_version == orig:
        neutral_version = "I am disappointed with this outcome and would like to see improvements."

    # 3) Positive/constructive rewrite: reframe as constructive feedback
    p = orig
    for k, v in positive_map.items():
        p = replace_word_preserve_case(p, k, v)
    # Add constructive ending if not present
    if not re.search(r"(suggest|recommend|could|should|please|would be good)", p, re.IGNORECASE):
        p = p.rstrip(".!") + ". I suggest reviewing the process and making improvements."
    positive_version = p

    # final cleanups: fix spacing
    soft_version = clean_text(soft_version)
    neutral_version = clean_text(neutral_version)
    positive_version = clean_text(positive_version)

    return [soft_version, neutral_version, positive_version]

# --- UI ---
st.markdown("Enter your post below and click **Analyze**.")
user_input = st.text_area("Your post", height=180)

col1, col2 = st.columns([1,1])
with col1:
    if st.button("Analyze"):
        text = clean_text(user_input)
        if not text:
            st.warning("Please enter text to analyze.")
        else:
            neg_pct, full = neg_percentage(text)
            st.markdown("### Results")
            st.write(f"- **Negativity:** {neg_pct:.2f}%")
            st.write(f"- **VADER scores:** {full}")

            if neg_pct > 50:
                st.error("❌ Blocked: Negativity is above 50%")
                st.info("You can still request rewrites to reduce negativity (choose below).")

                # show rewrite button
                if st.button("Request Rewrite (generate 3 options)"):
                    options = generate_rewrites(text)
                    st.markdown("### ✨ Rewrite options")
                    for i, opt in enumerate(options, start=1):
                        st.write(f"**Option {i}:** {opt}")
                        if st.button(f"Select Option {i}"):
                            st.success("Selected — final text:")
                            st.write(opt)
                            # Optionally allow copying or further edits
                            st.text_area("Final text (editable):", value=opt, height=120, key=f"final_{i}")
            else:
                st.success("✅ Allowed: Negativity is 50% or below.")
                # offer rewrite choice as optional
                if st.button("Rewrite to Reduce Negativity"):
                    options = generate_rewrites(text)
                    st.markdown("### ✨ Rewrite options")
                    for i, opt in enumerate(options, start=1):
                        st.write(f"**Option {i}:** {opt}")
                        if st.button(f"Select Option {i}"):
                            st.success("Selected — final text:")
                            st.write(opt)
                            st.text_area("Final text (editable):", value=opt, height=120, key=f"final_ok_{i}")

with col2:
    st.markdown("## Notes")
    st.write("- Limit for blocking is set to **50% negativity**.")
    st.write("- Rewrites are rule-based and aim to reduce harsh words and reframe the message.")
    
    st.write("- Use the Select button below any rewritten option to pick it as the final text.")

