import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

import re
import string
import zipfile
import numpy as np
import pandas as pd
import plotly.express as px
import gradio as gr
from underthesea import word_tokenize, text_normalize
import tf_keras as keras
from transformers import AutoTokenizer


# 1. AUTO-EXTRACT AND MODEL LOADING

def extract_bundle():
  
    bundle_name = "phobert_production_bundle.zip"
    target_dir = "model/phobert_bundle"
    
    if not os.path.exists(target_dir):
        if os.path.exists(bundle_name):
            print(f"[*] Extracting {bundle_name}...")
            with zipfile.ZipFile(bundle_name, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
        elif os.path.exists(f"model/{bundle_name}"):
            print(f"[*] Extracting model/{bundle_name}...")
            with zipfile.ZipFile(f"model/{bundle_name}", 'r') as zip_ref:
                zip_ref.extractall(target_dir)

def load_ai_model():

    extract_bundle()
    possible_paths = ["model/phobert_bundle", "phobert_bundle"]
    
    for path in possible_paths:
        keras_path = os.path.join(path, "keras_model")
        tok_path = os.path.join(path, "tokenizer")
        
        if not os.path.exists(keras_path):
            keras_path, tok_path = path, path

        if os.path.exists(keras_path) and os.path.exists(tok_path):
            try:
                tokenizer = AutoTokenizer.from_pretrained(tok_path)
                model = keras.models.load_model(keras_path, compile=False)
                print("[*] PhoBERT loaded successfully!")
                return model, tokenizer, None
            except Exception as e:
                return None, None, str(e)
                
    return None, None, 


MODEL, TOKENIZER, LOAD_ERROR = load_ai_model()


# 2. NLP PREPROCESSING

VIETNAMESE_ABBREVIATIONS = {
    "dc": "được", "đc": "được", "ok": "được", "k": "không", "ko": "không", 
    "kh": "không", "v": "vậy", "st": "sinh viên", "sv": "sinh viên", 
    "gv": "giảng viên", "bt": "bình thường", "qá": "quá", "e": "em", 
    "t": "tôi", "ae": "anh em", "tks": "cảm ơn", "thx": "cảm ơn", "gđ": "gia đình"
}

def clean_text(text):
  
    if not isinstance(text, str) or not text.strip(): return ""
    
    text = text.lower()
    for abbr, full_word in VIETNAMESE_ABBREVIATIONS.items():
        text = re.sub(rf"\b{abbr}\b", full_word, text)
        
    text = re.sub(r'[^\w\s,]', '', text)
    text = re.sub(r'(([a-z])\2{1,})+', r'\2', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    
    text = text_normalize(text)
    text = word_tokenize(text, format="text")
    return text


# 3. INFERENCE LOGIC

def predict_single(comment):
  
    if MODEL is None or TOKENIZER is None:
        raise gr.Error(f"System not ready: {LOAD_ERROR}")
    
    if not comment.strip():
        return {"Please enter some text": 1.0}
        
    cleaned_comment = clean_text(comment)
    
    encoded = TOKENIZER(text=cleaned_comment, padding='max_length', truncation=True, max_length=256, return_tensors='np')
    prediction = MODEL(np.array([cleaned_comment]))
    pred_arr = prediction.numpy()[0] if hasattr(prediction, 'numpy') else prediction[0]
    
    return {
        "🔴 Negative": float(pred_arr[0]),
        "⚪ Neutral": float(pred_arr[1]),
        "🟢 Positive": float(pred_arr[2])
    }

def process_batch_file(file_obj):
 
    if file_obj is None:
        raise gr.Error("Please upload a data file!")
        
    try:
        if file_obj.name.endswith('.csv'):
            df = pd.read_csv(file_obj.name)
        elif file_obj.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_obj.name)
        else:
            raise gr.Error("Only .csv or .xlsx files are supported.")

   
        text_col = None
        keywords = ['comment', 'text', 'nhận xét', 'nội dung', 'feedback']
        for col in df.columns:
            if any(kw in str(col).lower() for kw in keywords):
                text_col = col; break
        if not text_col:
            text_col = df.select_dtypes(include=['object', 'string']).columns[0]
            
        df['Preprocessed_Text'] = df[text_col].astype(str).apply(clean_text)
        
        results = []
        for text in df['Preprocessed_Text']:
            if not text.strip():
                results.append("Neutral"); continue
            
            pred = MODEL(np.array([text]))
            pred_arr = pred.numpy()[0] if hasattr(pred, 'numpy') else pred[0]
            idx = np.argmax(pred_arr)
            
            if idx == 0: results.append("Negative")
            elif idx == 1: results.append("Neutral")
            else: results.append("Positive")
            
        df['Predicted_Sentiment'] = results
        
    
        sentiment_counts = df['Predicted_Sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        color_map = {"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral": "#95a5a6"}
        
        fig_pie = px.pie(
            sentiment_counts, values='Count', names='Sentiment',
            title="Student Sentiment Distribution",
            color='Sentiment', color_discrete_map=color_map, hole=0.4
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label', hoverinfo='label+percent+value')
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

        return fig_pie, df, f"✅ Complete! Successfully analyzed {len(df)} comments from column '{text_col}'."
        
    except Exception as e:
        raise gr.Error(f"System error while processing the file: {str(e)}")


# 4. USER INTERFACE (UI/UX)

CUSTOM_CSS = """
.gradio-container { font-family: 'Inter', -apple-system, sans-serif !important; }
.header-box { text-align: center; padding: 2.5rem; background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%); color: white; border-radius: 12px; margin-bottom: 2rem; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); }
.header-box h1 { margin: 0; font-size: 2.8rem; font-weight: 800; color: white; letter-spacing: -0.02em; }
.header-box p { margin-top: 12px; font-size: 1.15rem; opacity: 0.9; font-weight: 300; }
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo"), css=CUSTOM_CSS, title="VietEduSent Dashboard") as app:
    
    gr.HTML("""
        <div class="header-box">
            <h1>VietEduSent Framework</h1>
            <p>A Production-Ready NLP Architecture for Vietnamese Sentiment Analysis</p>
        </div>
    """)
    
    with gr.Tabs():
        #  TAB 1: Real-time Analysis 
        with gr.Tab("📝 Single Analysis (Real-time)"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 💬 Enter Student Feedback")
                    input_text = gr.Textbox(show_label=False, lines=5, placeholder="Example: Giảng viên dạy rất nhiệt tình, slide bài giảng đẹp và cực kỳ dễ hiểu...")
                    with gr.Row():
                        btn_clear = gr.Button("Clear", size="lg")
                        btn_submit = gr.Button("Analyze Sentiment 🚀", variant="primary", size="lg")
                        
                    gr.Examples(
                        examples=[
                            ["Thầy dạy rất hăng say, sv thx thầy lắm."],
                            ["Slide quá khó hiểu, gv ko giải thích rõ ràng."],
                            ["Phòng học nóng, cơ sở vật chất xuống cấp làm giảm chất lượng học tập."]
                        ],
                        inputs=input_text, label="Sample Data (Click to try)"
                    )
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Prediction Confidence (PhoBERT)")
                    output_labels = gr.Label(num_top_classes=3, show_label=False)
            
            btn_submit.click(fn=predict_single, inputs=input_text, outputs=output_labels)
            btn_clear.click(fn=lambda: ("", None), inputs=None, outputs=[input_text, output_labels])

        #  TAB 2: Batch Processing 
        with gr.Tab("📂 Batch Processing"):
            gr.Markdown("Upload a `.csv` or `.xlsx` file containing student feedback. The model will process the entire list and extract an interactive dashboard.")
            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(label="Upload Data File", file_types=[".csv", ".xlsx"])
                    btn_batch = gr.Button("Start Processing ⚙️", variant="primary")
                    status_text = gr.Textbox(label="Status Log", interactive=False)
                with gr.Column(scale=2):
                    plot_output = gr.Plot(label="Statistical Distribution (Interactive Plotly)")
            
            gr.Markdown("### 📋 Extracted Data")
            table_output = gr.Dataframe(headers=["Original Feedback", "Preprocessed Text", "Predicted Label"], interactive=False)
            
            btn_batch.click(fn=process_batch_file, inputs=file_input, outputs=[plot_output, table_output, status_text])

        # TAB 3: Research Overview 
        with gr.Tab("ℹ️ About the Project"):
            gr.Markdown("""
            ### 🎓 Academic Information
            **VietEduSent** is a comprehensive Natural Language Processing (NLP) framework explicitly optimized for analyzing Vietnamese student feedback sentiment.
            
            - **Core Architecture:** Built upon Transformer models (PhoBERT-Base).
            - **Data Preprocessing:** Integrates a robust rule-based engine tailored for student-specific abbreviations alongside the standard `underthesea` tokenizer.
            - **End-to-End Pipeline:** Fully automates the workflow from text normalization and model inference to production-grade data visualization.
            """)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
