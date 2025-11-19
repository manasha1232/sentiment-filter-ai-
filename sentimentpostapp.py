# sentimentpostapp.py
import streamlit as st
import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
import os
from openai import OpenAI
from dotenv import load_dotenv

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

from nltk.tokenize import word_tokenize

# Load environment variables
load_dotenv()

# Configure the page
st.set_page_config(
    page_title="Negativity Analysis Platform",
    page_icon="📊",
    layout="centered"
)

# Custom CSS for professional lilac theme
st.markdown("""
<style>
    .main {
        background-color: #f8f4ff;
    }
    .stTextArea textarea {
        border: 2px solid #c8a2c8;
        border-radius: 8px;
        font-size: 14px;
    }
    .block-container {
        padding-top: 2rem;
    }
    .negative-alert {
        background-color: #ffe6e6;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #ff6b6b;
        margin: 1rem 0;
    }
    .positive-alert {
        background-color: #e6ffe6;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    .rewrite-option {
        background-color: #f0e6ff;
        padding: 1.5rem;
        border-radius: 8px;
        border: 2px solid #c8a2c8;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    .rewrite-option:hover {
        background-color: #e8daff;
        transform: translateY(-2px);
    }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e0d0e0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
    .header-section {
        background: linear-gradient(135deg, #c8a2c8, #9b59b6);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .analysis-section {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        border: 1px solid #e0d0e0;
        margin: 1rem 0;
    }
    .stButton button {
        background-color: #c8a2c8;
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: 500;
    }
    .stButton button:hover {
        background-color: #9b59b6;
        color: white;
    }
    .selected-rewrite {
        border: 3px solid #9b59b6;
        background-color: #e8daff;
    }
    .ai-rewrite {
        border: 2px dashed #9b59b6;
    }
</style>
""", unsafe_allow_html=True)

class EnhancedNegativityAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        
        # Enhanced negative word list with weights
        self.negative_lexicon = {
            'hate': 0.9, 'despise': 0.9, 'loathe': 0.9, 'terrible': 0.8, 'awful': 0.8,
            'horrible': 0.8, 'disgusting': 0.85, 'useless': 0.7, 'worthless': 0.8,
            'garbage': 0.8, 'trash': 0.7, 'shit': 0.9, 'crap': 0.7, 'fuck': 0.9,
            'damn': 0.6, 'hell': 0.6, 'stupid': 0.7, 'idiot': 0.8, 'moron': 0.8,
            'ridiculous': 0.6, 'pathetic': 0.8, 'disaster': 0.7, 'broken': 0.6,
            'furious': 0.8, 'angry': 0.7, 'pissed': 0.8, 'annoying': 0.6,
            'never': 0.5, 'worst': 0.9, 'horrendous': 0.9, 'appalling': 0.8,
            'unacceptable': 0.7, 'failure': 0.7, 'waste': 0.6, 'pointless': 0.7
        }
    
    def calculate_enhanced_negativity(self, text):
        """Enhanced negativity calculation combining VADER and custom lexicon"""
        text_lower = text.lower()
        
        # Get VADER score
        vader_scores = self.analyzer.polarity_scores(text)
        vader_negativity = vader_scores['neg'] * 100
        
        # Calculate lexicon-based negativity
        try:
            words = word_tokenize(text_lower)
        except:
            # Fallback to simple split if tokenization fails
            words = text_lower.split()
            
        lexicon_score = 0
        negative_word_count = 0
        
        for word in words:
            # Clean the word
            clean_word = re.sub(r'[^\w\s]', '', word)
            if clean_word in self.negative_lexicon:
                lexicon_score += self.negative_lexicon[clean_word]
                negative_word_count += 1
        
        # Normalize lexicon score to percentage
        if words:
            lexicon_negativity = (lexicon_score / len(words)) * 100
        else:
            lexicon_negativity = 0
        
        # Combine both scores (weighted average)
        final_negativity = (vader_negativity * 0.6) + (lexicon_negativity * 0.4)
        
        return min(final_negativity, 100), vader_scores, negative_word_count
    
    def analyze_sentence_level_negativity(self, text):
        """Analyze negativity at sentence level"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        sentence_analysis = []
        for sentence in sentences:
            if sentence:
                negativity, scores, _ = self.calculate_enhanced_negativity(sentence)
                sentence_analysis.append({
                    'sentence': sentence,
                    'negativity': negativity,
                    'compound': scores['compound']
                })
        
        return sentence_analysis
    
    def get_sentiment_label(self, negativity_score):
        """Get sentiment label based on negativity percentage"""
        if negativity_score < 20:
            return "Positive", "#4CAF50"
        elif negativity_score < 50:
            return "Neutral", "#FFA500"
        elif negativity_score < 75:
            return "Negative", "#FF6B6B"
        else:
            return "Highly Negative", "#DC143C"

class TextRewriter:
    def __init__(self):
        self.analyzer = EnhancedNegativityAnalyzer()
        self.client = None
        self.openai_available = False
        
        # Initialize OpenAI if API key is available
        if os.getenv('OPENAI_API_KEY'):
            try:
                self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                self.openai_available = True
            except Exception as e:
                st.warning(f"OpenAI initialization failed: {e}")
    
    def rewrite_with_openai(self, text):
        """Rewrites text to be positive and constructive using OpenAI"""
        if not self.openai_available:
            return "OpenAI API not available. Please check your API key."
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that specializes in rewriting text to be positive, professional, and constructive while preserving the original meaning. Always respond with just the rewritten text."},
                    {"role": "user", "content": f"Rewrite this text to be positive and constructive: '{text}'"}
                ],
                max_tokens=150,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error in OpenAI rewrite: {str(e)}"
    
    def rule_based_rewrite(self, text):
        """Rule-based text rewriting to reduce negativity"""
        replacement_patterns = {
            r'\b(hate|detest|despise)\b': 'dislike',
            r'\b(awful|terrible|horrible)\b': 'disappointing',
            r'\b(stupid|idiotic|dumb)\b': 'unwise',
            r'\b(shit|crap|garbage)\b': 'disappointing',
            r'\b(pissed|angry|furious)\b': 'frustrated',
            r'\b(useless|worthless|pointless)\b': 'ineffective',
            r'\b(mess|disaster|catastrophe)\b': 'challenging situation',
            r'\b(fail|screw up|mess up)\b': 'could be improved',
            r'\b(never|always|everyone|nobody)\b': 'often/many',
        }
        
        rewritten = text
        for pattern, replacement in replacement_patterns.items():
            rewritten = re.sub(pattern, replacement, rewritten, flags=re.IGNORECASE)
        
        return rewritten
    
    def structural_rewrite(self, text):
        """Rewrite by changing sentence structure to be more constructive"""
        structural_patterns = [
            (r'\b(I hate|I can\'t stand|I despise)\b', 'I would prefer'),
            (r'\b(This is|It\'s) (awful|terrible|horrible)\b', 'This could be improved'),
            (r'\b(You|They|Everyone) (always|never)\b', 'Sometimes'),
            (r'\b(It doesn\'t work|It\'s broken)\b', 'It could function better'),
            (r'\b(I\'m angry|I\'m pissed|I\'m furious)\b', 'I feel concerned'),
        ]
        
        rewritten = text
        for pattern, replacement in structural_patterns:
            rewritten = re.sub(pattern, replacement, rewritten, flags=re.IGNORECASE)
        
        # Add constructive framing
        if not any(phrase in rewritten.lower() for phrase in ['suggest', 'recommend', 'improve', 'better']):
            if rewritten and rewritten[-1] in ['.', '!', '?']:
                rewritten = rewritten[:-1] + ". I suggest some improvements could be made."
            else:
                rewritten += " I believe this could be improved."
        
        return rewritten
    
    def generate_rewrites(self, text):
        """Generate multiple rewritten versions"""
        rewrites = {}
        
        # Rule-based rewrite
        rule_based = self.rule_based_rewrite(text)
        rule_negativity, _, _ = self.analyzer.calculate_enhanced_negativity(rule_based)
        rewrites["Neutral Language"] = {
            "text": rule_based,
            "negativity": rule_negativity,
            "description": "Replaced harsh words with neutral alternatives"
        }
        
        # Structural rewrite
        structural = self.structural_rewrite(text)
        struct_negativity, _, _ = self.analyzer.calculate_enhanced_negativity(structural)
        rewrites["Constructive Feedback"] = {
            "text": structural,
            "negativity": struct_negativity,
            "description": "Reframed as constructive feedback with suggestions"
        }
        
        # AI-powered rewrite (if available)
        if self.openai_available:
            ai_rewrite = self.rewrite_with_openai(text)
            ai_negativity, _, _ = self.analyzer.calculate_enhanced_negativity(ai_rewrite)
            rewrites["AI Optimized"] = {
                "text": ai_rewrite,
                "negativity": ai_negativity,
                "description": "AI-powered positive reconstruction"
            }
        else:
            # Fallback soft rewrite
            soft = self.soft_rewrite(text)
            soft_negativity, _, _ = self.analyzer.calculate_enhanced_negativity(soft)
            rewrites["Softer Tone"] = {
                "text": soft,
                "negativity": soft_negativity,
                "description": "Maintained message with softened language"
            }
        
        return rewrites
    
    def soft_rewrite(self, text):
        """Soften the language while maintaining the core message"""
        softening_patterns = [
            (r'\b(hate)\b', 'strongly dislike'),
            (r'\b(awful)\b', 'quite disappointing'),
            (r'\b(terrible)\b', 'very disappointing'),
            (r'\b(stupid)\b', 'not the best approach'),
            (r'\b(angry)\b', 'concerned'),
            (r'\b(furious)\b', 'quite concerned'),
            (r'\b(useless)\b', 'not very effective'),
        ]
        
        rewritten = text
        for pattern, replacement in softening_patterns:
            rewritten = re.sub(pattern, replacement, rewritten, flags=re.IGNORECASE)
        
        return rewritten

def main():
    # Header Section
    st.markdown("""
    <div class="header-section">
        <h1 style="margin:0; font-size: 2.5rem;">Advanced Negativity Analysis Platform</h1>
        <p style="margin:0; font-size: 1.1rem; opacity: 0.9;">
        AI-powered sentiment analysis with intelligent rewriting
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize analyzers
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = EnhancedNegativityAnalyzer()
    if 'rewriter' not in st.session_state:
        st.session_state.rewriter = TextRewriter()
    if 'selected_rewrite' not in st.session_state:
        st.session_state.selected_rewrite = None
    
    # OpenAI status
    if st.session_state.rewriter.openai_available:
        st.success("✅ OpenAI integration active - AI rewriting enabled")
    else:
        st.warning("⚠️ OpenAI not configured - using rule-based rewriting only")
        st.info("To enable AI rewriting, create a .env file with your OPENAI_API_KEY")
    
    # Input Section
    st.markdown("### Analyze Your Text")
    user_text = st.text_area(
        "Enter your text below for sentiment analysis:",
        placeholder="Type your message here...",
        height=120,
        key="user_input"
    )
    
    if st.button("Analyze Sentiment", use_container_width=True):
        if user_text.strip():
            with st.spinner("Analyzing sentiment..."):
                # Perform enhanced analysis
                negativity_score, full_scores, negative_words = st.session_state.analyzer.calculate_enhanced_negativity(user_text)
                sentence_analysis = st.session_state.analyzer.analyze_sentence_level_negativity(user_text)
                sentiment_label, sentiment_color = st.session_state.analyzer.get_sentiment_label(negativity_score)
                
                # Store results in session state
                st.session_state.analysis_results = {
                    'original_text': user_text,
                    'negativity_score': negativity_score,
                    'full_scores': full_scores,
                    'sentence_analysis': sentence_analysis,
                    'sentiment_label': sentiment_label,
                    'sentiment_color': sentiment_color,
                    'negative_word_count': negative_words
                }
                
                # Generate rewrites if needed
                if negativity_score > 50:
                    st.session_state.rewrites = st.session_state.rewriter.generate_rewrites(user_text)
    
    # Display Results
    if 'analysis_results' in st.session_state:
        results = st.session_state.analysis_results
        
        st.markdown("---")
        st.markdown("### Analysis Results")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: {results['sentiment_color']}; margin:0;">{results['negativity_score']:.1f}%</h3>
                <p style="margin:0; color: #666;">Negativity Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: {results['sentiment_color']}; margin:0;">{results['sentiment_label']}</h3>
                <p style="margin:0; color: #666;">Sentiment</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            compound_color = "#4CAF50" if results['full_scores']['compound'] >= 0.05 else "#FF6B6B" if results['full_scores']['compound'] <= -0.05 else "#FFA500"
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: {compound_color}; margin:0;">{results['full_scores']['compound']:.2f}</h3>
                <p style="margin:0; color: #666;">Compound Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #FF6B6B; margin:0;">{results['negative_word_count']}</h3>
                <p style="margin:0; color: #666;">Negative Words</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Negativity Alert and Rewriting Options
        if results['negativity_score'] > 50:
            st.markdown(f"""
            <div class="negative-alert">
                <h4 style="color: #DC143C; margin:0;">⚠️ High Negativity Detected</h4>
                <p style="margin:0.5rem 0 0 0;">This text exceeds the 50% negativity threshold and may be blocked from posting.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### Rewriting Options")
            st.info("Select a rewritten version below to reduce negativity while maintaining your message.")
            
            if 'rewrites' in st.session_state:
                for i, (name, rewrite_data) in enumerate(st.session_state.rewrites.items()):
                    reduction = results['negativity_score'] - rewrite_data['negativity']
                    is_ai = "AI" in name
                    
                    with st.container():
                        css_class = "rewrite-option ai-rewrite" if is_ai else "rewrite-option"
                        if st.session_state.selected_rewrite == name:
                            css_class += " selected-rewrite"
                            
                        st.markdown(f"""
                        <div class="{css_class}">
                            <h4>{name} {"🤖" if is_ai else ""}</h4>
                            <p><strong>Description:</strong> {rewrite_data['description']}</p>
                            <p><strong>Negativity Reduction:</strong> <span style="color: #4CAF50;">-{reduction:.1f}%</span></p>
                            <p><strong>New Score:</strong> {rewrite_data['negativity']:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.write("**Rewritten Text:**")
                        st.write(rewrite_data['text'])
                        
                        if st.button(f"Select {name}", key=f"select_{i}"):
                            st.session_state.selected_rewrite = name
                            st.session_state.final_text = rewrite_data['text']
                            st.rerun()
                
                # Final selection and posting
                if st.session_state.selected_rewrite:
                    st.markdown("---")
                    st.markdown("### Final Selection")
                    st.success(f"✅ Selected: {st.session_state.selected_rewrite}")
                    st.text_area("Your approved text:", st.session_state.final_text, height=100)
                    
                    if st.button("Post Approved Text", type="primary", use_container_width=True):
                        st.balloons()
                        st.success("Your message has been posted successfully!")
                        # Reset for new analysis
                        st.session_state.selected_rewrite = None
                        if 'final_text' in st.session_state:
                            del st.session_state.final_text
        else:
            st.markdown(f"""
            <div class="positive-alert">
                <h4 style="color: #4CAF50; margin:0;">✓ Acceptable Negativity Level</h4>
                <p style="margin:0.5rem 0 0 0;">This text meets the platform guidelines and can be posted.</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Post Text", type="primary", use_container_width=True):
                st.balloons()
                st.success("Your message has been posted successfully!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>Advanced Negativity Analysis Platform • Powered by VADER + OpenAI • Professional Communication Assistant</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
