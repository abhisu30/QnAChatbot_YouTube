#Uses a different technique as compared to V1. Uses Metadata Replacement + Node Sentence Window as demonstrated in this document - https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/MetadataReplacementDemo.html

import os
import streamlit as st
import json
import chromadb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.vector_stores.chroma import ChromaVectorStore
from pytube import YouTube
from docx import Document
import whisper
import re

# Function to validate YouTube URL
def validate_youtube_url(url):
    youtube_regex = (
        r'(https?://)?(www\.)?'
        '(youtube|youtu|youtube-nocookie)\.(com|be)/'
        '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')
    return re.match(youtube_regex, url)

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="/cdb")

# Set up Streamlit page
st.set_page_config(page_title="Chat With Youtube Videos", page_icon="📚", layout="centered")
st.title("QnA Chatbot for Youtube Videos 💬")

# Sidebar for input
with st.sidebar:
    youtube_url = st.text_input("YouTube URL")
    openai_api_key = st.text_input("OpenAI API Key:", type="password")
    process_button = st.button("Process", disabled=(not youtube_url or not openai_api_key))

# Function to download audio from YouTube and transcribe it
def process_youtube_url(url):
    yt = YouTube(url)
    video = yt.streams.filter(only_audio=True).first()
    audio_filename = video.download(output_path='Data/Audio', filename_prefix='transcript_')
    model = whisper.load_model("base")
    result = model.transcribe(audio_filename)
    text = result['text']
    doc_filename = os.path.join('Data', 'Docs', f"{video.title}.docx")
    doc = Document()
    doc.add_paragraph(text)
    doc.save(doc_filename)
    return doc_filename

# Process YouTube URL and update the document path
if process_button:
    if not validate_youtube_url(youtube_url):
        st.error("Invalid YouTube URL.")
    else:
        try:
            doc_path = process_youtube_url(youtube_url)
            st.session_state['doc_path'] = doc_path
            st.success("Video has processed and ready for chat.")
        except Exception as e:
            st.error(f"Error processing the video: {e}")

# Initialize the LLM and embedding model
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.5, api_key=openai_api_key)
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create the sentence window node parser with default settings
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=5,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)

# Function to load data and initialize chat engine
@st.cache_resource(show_spinner=False)
def load_data(doc_path):
    if doc_path:
        with st.spinner(text="Chatbot Initializing – hang tight! This should take a moment."):
            documents = SimpleDirectoryReader(input_dir=os.path.dirname(doc_path), required_exts=['.docx']).load_data()
            nodes = node_parser.get_nodes_from_documents(documents)
            vector_store = ChromaVectorStore(chroma_client.get_or_create_collection("videodb"))
            index = VectorStoreIndex(nodes, vector_store=vector_store, embed_model=embed_model)
            return index
    return None

# Load data and initialize chat engine
if "doc_path" in st.session_state:
    index = load_data(st.session_state['doc_path'])
    if index:
        chat_engine = index.as_chat_engine(
            chat_mode="openai",
            llm=llm,
            verbose=True,
            similarity_top_k=5,
            node_postprocessors=[
                MetadataReplacementPostProcessor(target_metadata_key="window")
            ],
        )

# Initialize the chat messages history
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about the Video!"}
    ]

# Display chat messages and handle user input
if "chat_engine" in locals():
    if prompt := st.chat_input("Your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                Final_prompt = (
                    "You are answering questions of the user about a Video Transcript. You will refer to the chat history if required. Chat History: "
                    + chat_history + "\n User: " + prompt)
                try:
                    response = chat_engine.chat(Final_prompt)
                    st.write(response.response)
                    message = {"role": "assistant", "content": response.response}
                    st.session_state.messages.append(message)
                except Exception:
                    st.error("AI not responding.")
else:
    st.warning("Please process a Youtube Video first.")

# Ensure necessary directories exist
if not os.path.exists('Data/Audio'):
    os.makedirs('Data/Audio')

# Refresh chat button
if st.button("Refresh Chat"):
    st.session_state.messages = []
    st.experimental_rerun()
