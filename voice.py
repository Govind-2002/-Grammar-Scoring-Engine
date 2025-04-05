# voice.py
import os
import torch
import librosa
import streamlit as st
import language_tool_python
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Configuration
MAX_FILE_SIZE_MB = 200
SAMPLING_RATE = 16000

@st.cache_resource
def load_models():
    """Initialize models with remote LanguageTool configuration"""
    torch.set_num_threads(1)
    return {
        'stt': pipeline(
            "automatic-speech-recognition", 
            "openai/whisper-medium",
            device="cpu"
        ),
        'grammar_tool': language_tool_python.LanguageTool(
            'en-US',
            remote_server='https://api.languagetool.org'
        ),
        'grader': AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-CoLA"),
        'tokenizer': AutoTokenizer.from_pretrained("textattack/bert-base-uncased-CoLA"),
        'corrector': pipeline(
            "text2text-generation",
            "pszemraj/flan-t5-large-grammar-synthesis",
            device="cpu"
        )
    }

def calculate_score(text, matches, grader, tokenizer):
    """Calculate grammar score (0-100) using hybrid approach"""
    # Rule-based error score (70% weight)
    error_score = min(len(matches)/10, 1.0)  # Cap at 10 errors
    
    # Model-based fluency score (30% weight)
    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
            return_attention_mask=True
        )
        with torch.no_grad():
            outputs = grader(**inputs)
        
        # Handle model output based on its format
        if outputs.logits.shape[1] == 2:  # Binary classification format
            probs = torch.softmax(outputs.logits, dim=1)
            fluency_prob = probs[0][1].item()  # Probability of acceptable grammar
        else:  # Regression format
            fluency_prob = torch.sigmoid(outputs.logits).item()
    
    except Exception as e:
        st.error(f"Fluency analysis failed: {str(e)}")
        fluency_prob = 0.7  # Default value on error
    
    return round((0.7*(1 - error_score) + 0.3*fluency_prob) * 100, 1)

def main():
    st.set_page_config(
        page_title="Grammar Scoring Engine",
        page_icon="✅",
        layout="centered"
    )
    st.title("Grammar Scoring Engine")
    
    models = load_models()
    
    # File upload with size limit
    audio_file = st.file_uploader(
        f"Upload audio file (WAV/MP3) - Max {MAX_FILE_SIZE_MB}MB",
        type=["wav", "mp3"],
        accept_multiple_files=False
    )
    
    if audio_file:
        # Validate file size
        if audio_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            st.error(f"File size exceeds {MAX_FILE_SIZE_MB}MB limit")
            return
            
        temp_path = f"temp_{audio_file.name}"
        try:
            # Save and process file
            with st.spinner("Processing audio..."):
                with open(temp_path, "wb") as f:
                    f.write(audio_file.getbuffer())
                
                # Transcribe audio
                audio = librosa.load(temp_path, sr=SAMPLING_RATE)[0]
                transcript = models['stt']({
                    "raw": audio, 
                    "sampling_rate": SAMPLING_RATE
                })['text']
            
            # Analyze grammar
            with st.spinner("Analyzing text..."):
                matches = models['grammar_tool'].check(transcript)
                score = calculate_score(
                    transcript, 
                    matches, 
                    models['grader'], 
                    models['tokenizer']
                )
                
                # Generate correction
                corrected = models['corrector'](
                    f"grammar: {transcript}",
                    max_length=512,
                    num_beams=3
                )[0]['generated_text']
            
            # Display results
            st.subheader(f"Grammar Score: {score}/100")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("### Original Text")
                st.code(transcript)
                
                if matches:
                    st.markdown("### Top Issues")
                    for match in matches[:3]:
                        st.error(f"**{match.ruleId}**: {match.message}")
                else:
                    st.success("No grammatical issues found!")
            
            with col2:
                st.markdown("### Improved Version")
                st.success(corrected)
            
        except Exception as e:
            st.error(f"Processing failed: {str(e)}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == "__main__":
    main()