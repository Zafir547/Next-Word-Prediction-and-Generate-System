import os
import time
import requests
import streamlit as st
from typing import List, Dict

# CONFIGURATION - DEPLOYMENT READY

# Get API URL from environment or Streamlit secrets
API_BASE_URL = os.getenv("API_BASE_URL")

# Try to load from Streamlit secrets (for cloud deployment)
try:
    if hasattr(st, 'secrets') and "API_BASE_URL" in st.secrets:
        API_BASE_URL = st.secrets["API_BASE_URL"]
except:
    pass

# Page Configuration
st.set_page_config(
    page_title="Next Word Prediction",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session State Initialization
if "prediction_input" not in st.session_state:
    st.session_state.prediction_input = ""

if "generated_prompt" not in st.session_state:
    st.session_state.generated_prompt = ""


# Callback Functions
def set_example_text(text):
    st.session_state.prediction_input = text

def set_prompt_text(text):
    st.session_state.generated_prompt = text

def clear_prompt():
    st.session_state.generated_prompt = ""


# API Functions
def check_api_health():
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return r.status_code == 200, r.json()
    except Exception as e:
        return False, {"error": str(e)}

def predict_next_word(text, top_k, temperature):
    try:
        start_time = time.time()

        r = requests.post(
            f"{API_BASE_URL}/predict",
            json={"text": text, "top_k": top_k, "temperature": temperature},
            timeout=10
        )

        end_time = time.time()
        r.raise_for_status()

        data = r.json()
        data["total_request_time_ms"] = round((end_time - start_time) * 1000, 2)
        return data
        
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

def generate_text(prompt, length, temperature, top_k):
    try:
        start_time = time.time()

        r = requests.post(
            f"{API_BASE_URL}/generate",
            json={
                "prompt": prompt,
                "length": length,
                "temperature": temperature,
                "top_k": top_k
            },
            timeout=15
        )

        end_time = time.time()
        r.raise_for_status()

        data = r.json()
        data["total_request_time_ms"] = round((end_time - start_time) * 1000, 2)
        return data
       
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

# CSS LOADING WITH FALLBACK
def load_css(file_path: str = "assets/style.css"):
    """Load CSS with inline fallback if file not found"""
    try:
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        # Inline CSS fallback
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
        }
        
        .prediction-item {
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
            color: white;
        }
        
        .prediction-high {
            background: #667eea;
        }
        
        .prediction-medium {
            background: #764ba2;
        }
        
        .prediction-low {
            background: #f093fb;
        }
        
        .prediction-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .prediction-word {
            font-size: 1.3rem;
            font-weight: bold;
        }
        
        .prediction-prob {
            font-size: 1.2rem;
        }
        
        .progress-bar {
            background: rgba(255, 255, 255, 0.3);
            height: 8px;
            border-radius: 5px;
            margin-top: 0.5rem;
        }
        
        .progress-fill {
            background: white;
            height: 100%;
            border-radius: 5px;
        }
        
        .generated-text-box {
            background: #f0f2f6;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #667eea;
            margin: 1rem 0;
        }
        
        .stats-box {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            text-align: center;
        }
        
        .deployment-info {
            background: #fff3cd;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #ffc107;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)

# Load CSS
load_css()

# HELPER FUNCTIONS
def display_prediction(predictions: List[Dict]):
    """Display predictions with visual styling"""
    for i, pred in enumerate(predictions, 1):
        word = pred['word']
        prob = pred['probability'] * 100

        # Decide class based on probability
        if prob > 70:
            level_class = "prediction-high"
        elif prob > 40:
            level_class = "prediction-medium"
        else:
            level_class = "prediction-low"

        st.markdown(f"""
        <div class="prediction-item {level_class}">
            <div class="prediction-row">
                <span class="prediction-word">{i}. {word}</span>
                <span class="prediction-prob">{prob:.2f}%</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {prob}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def show_deployment_info():
    """Show deployment configuration info"""
    st.markdown(f"""
    <div class="deployment-info">
        <strong>🌐 API Configuration</strong><br>
        Current API URL: <code>{API_BASE_URL}</code><br>
        <small>Configure in Streamlit secrets for cloud deployment</small>
    </div>
    """, unsafe_allow_html=True)

# MAIN APPLICATION

def main():
    # Header
    st.sidebar.markdown('<h1 class="main-header">Next Word Prediction and Generate System</h1>', unsafe_allow_html=True)
    st.sidebar.markdown("---")

    # Check API health
    is_healthy, health_data = check_api_health()

    if not is_healthy:
        st.error("⚠️ Backend API is not running or not reachable.")
        
        # Show current configuration
        show_deployment_info()
        
        # Show connection details
        with st.expander("🔍 Connection Details"):
            st.write(f"**Attempting to connect to:** `{API_BASE_URL}`")
            st.write(f"**Error:** {health_data.get('error', 'Connection timeout')}")
            st.info("""
            **For Local Development:**
            ```bash
            cd backend
            python app.py
            ```
            
            **For Cloud Deployment:**
            1. Deploy backend to Render.com
            2. Add API URL to Streamlit secrets:
               ```toml
               API_BASE_URL = "https://your-backend.onrender.com"
               ```
            """)
        
        st.stop()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # API Status
        st.success("✅ API Connected")
        if health_data and 'error' not in health_data:
            st.info(f"📊 Vocab Size: {health_data.get('vocab_size', 'N/A')}")
            st.info(f"💻 Device: {health_data.get('device', 'N/A')}")
            
            # Show API URL in expander
            with st.expander("🌐 API Info"):
                st.code(API_BASE_URL)

        st.markdown("---")

        # Mode Selection
        mode = st.radio(
            "Select Mode:",
            ["🎯 Next Word Prediction", "✨ Text Generation"],
            index=0
        )

        st.markdown("---")

        # Parameters
        st.subheader("🎛️ Model Parameters")

        if mode == "🎯 Next Word Prediction":
            top_k = st.slider("Number of Predictions", 1, 10, 5)
            temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 0.1)
            st.caption("Higher temperature = more diverse predictions")
        else:
            gen_length = st.slider("Generation Length", 10, 50, 25)
            temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 0.1)
            top_k_gen = st.slider("Top-K Sampling", 5, 20, 10)
            st.caption("Higher values = more creative text")

        st.markdown("---")

        # About
        with st.expander("ℹ️ About"):
            st.write("""
            **Next Word Prediction System**
            
            This system uses an LSTM neural network trained on Pakistani Rupee dataset to predict the next word in a sequence.
            
            **Features:**
            - Real-time predictions
            - Text generation
            - Adjustable parameters
            - RESTful API backend
            
            **Technologies:**
            - Backend: FastAPI
            - Frontend: Streamlit
            - Model: LSTM (PyTorch)
            
            **Performance:**
            - Validation Accuracy: 94.13%
            - Perplexity: 1.92
            - Response Time: <1s
            """)

    # MAIN CONTENT AREA
    if mode == "🎯 Next Word Prediction":
        st.header("🎯 Next Word Prediction")
        st.write("Enter some text and get predictions for the next word!")

        col1, col2 = st.columns([3, 1])

        with col1:
            input_text = st.text_input(
                "Enter your text:",
                placeholder="Type something... e.g., 'The Pakistani rupee'",
                key="prediction_input"
            )

        with col2:
            predict_button = st.button("🔮 Predict", use_container_width=True)

        if predict_button and input_text:
            with st.spinner("Predicting..."):
                result = predict_next_word(input_text, top_k, temperature)

                if result and result.get('success'):
                    st.success("✅ Predictions ready!")

                    # Display input
                    st.markdown("### 📝 Your Input:")
                    st.info(f"**{input_text}**")

                    # Display predictions
                    st.markdown("### 🎯 Top Predictions:")
                    display_prediction(result['predictions'])

                    # Stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown('<div class="stats-box"><h3>📊</h3><p>Top K</p><h2>{}</h2></div>'.format(top_k), unsafe_allow_html=True)
                    with col2:
                        st.markdown('<div class="stats-box"><h3>🌡️</h3><p>Temperature</p><h2>{:.1f}</h2></div>'.format(temperature), unsafe_allow_html=True)
                    with col3:
                        top_prob = result['predictions'][0]['probability'] * 100
                        st.markdown('<div class="stats-box"><h3>🎯</h3><p>Confidence</p><h2>{:.1f}%</h2></div>'.format(top_prob), unsafe_allow_html=True)
                    
                    # Performance info
                    if 'total_request_time_ms' in result:
                        st.caption(f"⚡ Response time: {result['total_request_time_ms']:.0f}ms")

        elif predict_button:
            st.warning("⚠️ Please enter some text first!")

        # Examples
        st.markdown("---")
        st.markdown("### 💡 Try These Examples:")

        examples = [
            "The Pakistani rupee",
            "State Bank of",
            "In 1948 coins were",
            "The word rupiya is",
            "Reserve Bank of"
        ]

        cols = st.columns(len(examples))

        for i, example in enumerate(examples):
            cols[i].button(
                example,
                key=f"ex_{i}",
                on_click=set_example_text,
                args=(example,),
                use_container_width=True
            )

    else:   # Text Generation Mode
        st.header("✨ Text Generation")
        st.write("Start with a prompt and let the model generate text!")

        prompt = st.text_area(
            "Enter your prompt:",
            placeholder="Type a starting phrase... e.g., 'The Pakistani rupee was'",
            height=100,
            key="generated_prompt"
        )

        col1, col2 = st.columns([1, 1])
        
        with col1:
            generated_button = st.button("✨ Generate Text", use_container_width=True)

        with col2:
            st.button("🔄 Clear", on_click=clear_prompt, use_container_width=True)

        if generated_button and prompt:
            with st.spinner("Generating text..."):
                result = generate_text(prompt, gen_length, temperature, top_k_gen)

                if result and result.get('success'):
                    st.success("✅ Text generated!")

                    # Display generated text
                    st.markdown("### 📄 Generated Text:")
                    st.markdown(f"""
                    <div class="generated-text-box">
                        <p style="font-size: 1.2rem; line-height: 1.8; margin: 0;">
                            {result['generated_text']}
                        </p>            
                    </div>
                    """, unsafe_allow_html=True)

                    # Stats
                    col1, col2, col3 = st.columns(3)
                    word_count = len(result['generated_text'].split())
                    with col1:
                        st.markdown('<div class="stats-box"><h3>📝</h3><p>Words</p><h2>{}</h2></div>'.format(word_count), unsafe_allow_html=True)
                    with col2:
                        st.markdown('<div class="stats-box"><h3>🌡️</h3><p>Temperature</p><h2>{:.1f}</h2></div>'.format(temperature), unsafe_allow_html=True)
                    with col3:
                        st.markdown('<div class="stats-box"><h3>🎲</h3><p>Top-K</p><h2>{}</h2></div>'.format(top_k_gen), unsafe_allow_html=True)
                    
                    # Performance info
                    if 'total_request_time_ms' in result:
                        st.caption(f"⚡ Generation time: {result['total_request_time_ms']:.0f}ms")

                    # Download button
                    st.download_button(
                        label="📥 Download Text",
                        data=result['generated_text'],
                        file_name="generated_text.txt",
                        mime="text/plain"
                    )

        elif generated_button:
            st.warning("⚠️ Please enter a prompt first!")

        # Examples
        st.markdown("---")
        st.markdown("### 💡 Try These Prompts:")

        prompts = [
            "The Pakistani rupee was",
            "In 1948",
            "The State Bank of Pakistan",
            "Coins were introduced"
        ]

        cols = st.columns(len(prompts))

        for i, p in enumerate(prompts):
            cols[i].button(
                p,
                key=f"prompt_{i}",
                on_click=set_prompt_text,
                args=(p,),
                use_container_width=True
            )

# RUN APP
if __name__ == "__main__":
    main()
