import streamlit as st
import easyocr
import numpy as np
from PIL import Image
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="وکیل هوشمند", layout="wide")
st.title("⚖️ وکیل هوشمند (تحلیل تصویر)")

with st.sidebar:
    groq_api = st.text_input("Groq API Key را وارد کنید", type="password")
    reset = st.button("پاکسازی سشن")
    st.info("می‌توانید API را از سایت console.groq.com دریافت کنید")
if reset:
    st.session_state.clear()

@st.cache_resource
def load_models():
    reader = easyocr.Reader(['fa', 'en'], gpu=False)
    embeddings = None
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        embeddings = None
    return reader, embeddings

reader, embeddings = load_models()

uploaded_file = st.file_uploader("تصویر مدرک یا نامه حقوقی را انتخاب کنید", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    w, h = image.size
    max_side = 2000
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        image = image.resize((int(w * scale), int(h * scale)))
    st.image(image, caption="تصویر آپلود شده", width=400)

    with st.spinner("در حال استخراج متن از عکس (OCR)..."):
        try:
            image_np = np.array(image)
            result = reader.readtext(image_np)
            full_text = " ".join([x[1] for x in result]).strip()
        except Exception:
            st.error("استخراج متن با خطا مواجه شد.")
            st.stop()
        st.session_state.full_text = full_text

        docs = [Document(page_content=full_text)]
        st.session_state.vectorstore = None
        if embeddings is not None and full_text:
            try:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = [d.page_content for d in text_splitter.split_documents(docs)]
                st.session_state.vectorstore = FAISS.from_texts(chunks, embeddings)
            except Exception:
                st.session_state.vectorstore = None
        st.success("پردازش تصویر به پایان رسید!")

if "full_text" in st.session_state:
    st.divider()
    user_question = st.text_input("سوال حقوقی خود را در مورد این متن بپرسید:")
    if user_question:
        if not groq_api:
            st.warning("لطفاً ابتدا API Key را وارد کنید.")
            st.stop()
        with st.spinner("در حال تحلیل و پاسخگویی..."):
            try:
                llm = ChatGroq(temperature=0, groq_api_key=groq_api, model_name="llama3-8b-8192")
                template = "به عنوان یک وکیل خبره با توجه به متن زیر پاسخ دقیق و کوتاه بده:\nمتن: {context}\nسوال: {input}"
                prompt = ChatPromptTemplate.from_template(template)
                if st.session_state.get("vectorstore"):
                    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 2})
                    chain = ({"context": retriever, "input": RunnablePassthrough()} | prompt | llm | StrOutputParser())
                    answer = chain.invoke(user_question)
                else:
                    chain = (prompt | llm | StrOutputParser())
                    answer = chain.invoke({"context": st.session_state.full_text, "input": user_question})
            except Exception:
                st.error("حین تحلیل خطایی رخ داد.")
                st.stop()
        st.write("### 🤖 پاسخ وکیل:")
        st.info(answer)

with st.expander("مشاهده متن استخراج شده از تصویر"):
    if "full_text" in st.session_state:
        st.write(st.session_state.full_text)
    else:
        st.write("هنوز فایلی پردازش نشده است.")

