import re
import streamlit as st

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)

from langchain_core.output_parsers import StrOutputParser


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="YouTube AI Chatbot",
    page_icon="🎥",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #FF4B4B;
}

.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 30px;
}

.stTextInput > div > div > input {
    background-color: #262730;
    color: white;
}

.user-msg {
    background-color: #1565C0;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 10px;
    color: white;
}

.bot-msg {
    background-color: #2E7D32;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 20px;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="title">🎥 YouTube AI Chatbot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Ask Questions From Any YouTube Video</div>',
    unsafe_allow_html=True
)


# =========================================================
# AZURE OPENAI CONFIG
# =========================================================

AZURE_ENDPOINT = ""
API_KEY = ""

EMBEDDING_DEPLOYMENT = "text-embedding-3-small"
CHAT_DEPLOYMENT = "gpt-4o"

API_VERSION = "2024-12-01-preview"


# =========================================================
# SESSION STATE
# =========================================================

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# =========================================================
# FUNCTIONS
# =========================================================

def extract_video_id(url):

    """
    Extract YouTube Video ID from URL
    """

    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"

    match = re.search(pattern, url)

    if match:
        return match.group(1)

    return None


def get_transcript(video_id):

    """
    Get YouTube Transcript
    """

    try:

        ytt = YouTubeTranscriptApi()

        transcript_list = ytt.fetch(video_id)

        transcript = " ".join(
            chunk.text for chunk in transcript_list
        )

        return transcript

    except TranscriptsDisabled:

        st.error("No captions available for this video.")

        return None

    except Exception as e:

        st.error(f"Error: {e}")

        return None


def create_vectorstore(transcript):

    """
    Create FAISS Vector Store
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    documents = text_splitter.create_documents([transcript])

    embedding_model = AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=API_KEY,
        azure_deployment=EMBEDDING_DEPLOYMENT,
        openai_api_version=API_VERSION
    )

    vectorstore = FAISS.from_documents(
        documents,
        embedding_model
    )

    return vectorstore


def get_llm():

    """
    Load GPT Model
    """

    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        azure_deployment=CHAT_DEPLOYMENT,
        openai_api_version=API_VERSION,
        api_key=API_KEY,
        temperature=0.7
    )

    return llm


def format_docs(retrieved_docs):

    """
    Convert docs into text
    """

    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    return context


def ask_question(question):

    """
    Ask Question from YouTube Transcript
    """

    retriever = st.session_state.vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    llm = get_llm()

    prompt = PromptTemplate(
        template="""
You are a helpful assistant.

Answer ONLY from the provided transcript context.

If the context is insufficient, say:
"I don't know based on the video."

Transcript Context:
{context}

Question:
{question}

Answer:
""",
        input_variables=["context", "question"]
    )

    parallel_chain = RunnableParallel({

        "context": retriever | RunnableLambda(format_docs),

        "question": RunnablePassthrough()

    })

    parser = StrOutputParser()

    main_chain = parallel_chain | prompt | llm | parser

    response = main_chain.invoke(question)

    return response


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📌 Instructions")

    st.write("""
1. Paste YouTube URL  
2. Click Process Video  
3. Ask Questions  
4. Get AI Answers
    """)

    st.write("---")

    youtube_url = st.text_input(
        "Enter YouTube URL"
    )

    process_btn = st.button(
        "🚀 Process Video"
    )


# =========================================================
# PROCESS VIDEO
# =========================================================

if process_btn:

    if youtube_url == "":

        st.warning("Please enter YouTube URL")

    else:

        with st.spinner("Processing Video..."):

            video_id = extract_video_id(youtube_url)

            if not video_id:

                st.error("Invalid YouTube URL")

            else:

                transcript = get_transcript(video_id)

                if transcript:

                    vectorstore = create_vectorstore(
                        transcript
                    )

                    st.session_state.vectorstore = vectorstore

                    st.success(
                        "Video Processed Successfully ✅"
                    )

                    st.video(youtube_url)


# =========================================================
# CHAT SECTION
# =========================================================

st.write("---")

st.subheader("💬 Chat with Video")

question = st.text_input(
    "Ask your question"
)

ask_btn = st.button("Generate")


if ask_btn:

    if st.session_state.vectorstore is None:

        st.warning("Please process a video first.")

    elif question == "":

        st.warning("Please enter a question.")

    else:

        with st.spinner("Generating Answer..."):

            answer = ask_question(question)

            st.session_state.chat_history.append(
                ("USER", question)
            )

            st.session_state.chat_history.append(
                ("BOT", answer)
            )


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for role, message in st.session_state.chat_history:

    if role == "USER":

        st.markdown(
            f'<div class="user-msg"><b>You:</b><br>{message}</div>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f'<div class="bot-msg"><b>Bot:</b><br>{message}</div>',
            unsafe_allow_html=True
        )