import streamlit as st
from pytube import YouTube
from haystack.nodes import PromptNode, PromptModel
from haystack.nodes.audio import WhisperTranscriber
from haystack.pipelines import Pipeline
from model_add import LlamaCPPInvocationLayer
import time
import base64

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="YouTube Summarizer",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- CSS STYLES ----
st.markdown("""
    <style>
        .main {
            background-color: #f4f6f8;
        }
        h1 {
            color: #ff7f50;
            text-align: center;
        }
        h2, h3 {
            color: #4682b4;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            height: 3em;
            width: 100%;
        }
        .stDownloadButton>button {
            background-color: #008CBA;
            color: white;
            border-radius: 8px;
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

# ---- FUNCTIONS ----
def download_video(url):
    yt = YouTube(url)
    video = yt.streams.filter(abr='160kbps').last()
    return video.download()

def initialize_model(full_path):
    return PromptModel(
        model_name_or_path=full_path,
        invocation_layer_class=LlamaCPPInvocationLayer,
        use_gpu=False,
        max_length=512
    )

def initialize_prompt_node(model):
    summary_prompt = "deepset/summarization"
    return PromptNode(model_name_or_path=model, default_prompt_template=summary_prompt, use_gpu=False)

def transcribe_audio(file_path, prompt_node):
    whisper = WhisperTranscriber()
    pipeline = Pipeline()
    pipeline.add_node(component=whisper, name="whisper", inputs=["File"])
    pipeline.add_node(component=prompt_node, name="prompt", inputs=["whisper"])
    return pipeline.run(file_paths=[file_path])

# ---- APP ----
def main():
    st.title("🎥 YouTube Video Summarizer")
    st.subheader('✨ Built with LLaMA 2 🦙, Haystack, and Streamlit')

    with st.expander("ℹ️ About this App"):
        st.info("This app lets you summarize YouTube videos with AI.\n"
                "Just paste a YouTube URL, and get a concise AI-generated summary.")

    youtube_url = st.text_input("🔗 Paste YouTube URL here")

    col_input1, col_input2 = st.columns([1,1])
    with col_input1:
        submit = st.button("🚀 Summarize Video")
    with col_input2:
        clear = st.button("🗑️ Clear")

    if clear:
        st.experimental_rerun()

    if submit and youtube_url:
        with st.spinner("📥 Downloading video... Please wait."):
            start_time = time.time()
            file_path = download_video(youtube_url)

        with st.spinner("🤖 Initializing model..."):
            full_path = "llama-2-7b-32k-instruct.Q4_K_S.gguf"
            model = initialize_model(full_path)
            prompt_node = initialize_prompt_node(model)

        with st.spinner("🎤 Transcribing and summarizing..."):
            output = transcribe_audio(file_path, prompt_node)
            elapsed_time = time.time() - start_time

        col1, col2 = st.columns([1, 1])

        with col1:
            st.video(youtube_url)

        with col2:
            st.header("📜 Video Summary")
            summary_text = output["results"][0].split("\n\n[INST]")[0]
            st.success(summary_text)
            st.write(f"⏱️ Time taken: `{elapsed_time:.2f} seconds`")

            # Download button for summary
            b64 = base64.b64encode(summary_text.encode()).decode()
            href = f'<a href="data:file/txt;base64,{b64}" download="summary.txt">📥 Download Summary</a>'
            st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
