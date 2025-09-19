import streamlit as st
from main import multimodal_pdf_rag_pipeline

st.set_page_config(page_title="Multimodal RAG", layout="wide")

st.title("Multimodal RAG Query Interface")
st.write("Ask questions about your PDF (text + images).")

# Input box
query = st.text_input("Enter your query:")

# Submit button
if st.button("Ask"):
    if query.strip():
        with st.spinner("Retrieving answer..."):
            answer = multimodal_pdf_rag_pipeline(query)
        st.subheader("Answer:")
        st.write(answer)
    else:
        st.warning("!Please enter a query first.")
