import fitz  # PyMuPDF
from langchain_core.documents import Document
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
from langchain.schema.messages import HumanMessage
from sklearn.metrics.pairwise import cosine_similarity
import io
import base64
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

# ======================================
# Load environment variables (optional)
# ======================================
load_dotenv()

# ======================================
# Initialize CLIP model
# ======================================
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

# ======================================
# Embedding functions
# ======================================
def embed_text(text):
    inputs = clip_processor(
        text=text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    )
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()


def embed_image(image_data):
    if isinstance(image_data, str):
        image = Image.open(image_data).convert("RGB")
    else:
        image = image_data

    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()


# ======================================
# Process PDF
# ======================================
pdf_path = "IF10244.pdf"
doc = fitz.open(pdf_path)

all_docs = []
all_embeddings = []
image_data_store = {}

splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)

for i, page in enumerate(doc):
    # --- Process text ---
    text = page.get_text()
    if text.strip():
        temp_doc = Document(page_content=text, metadata={"page": i, "type": "text"})
        chunks = splitter.split_documents([temp_doc])
        for chunk in chunks:
            embedding = embed_text(chunk.page_content)
            all_embeddings.append(embedding)
            all_docs.append(chunk)

    # --- Process images ---
    for img_index, img in enumerate(page.get_images(full=True)):
        try:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_id = f"page_{i}_img_{img_index}"

            # Store base64 for reference
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            image_data_store[image_id] = img_base64

            # Embed image
            embedding = embed_image(pil_image)
            all_embeddings.append(embedding)

            image_doc = Document(
                page_content=f"[Image: {image_id}]",
                metadata={"page": i, "type": "image", "image_id": image_id}
            )
            all_docs.append(image_doc)

        except Exception as e:
            print(f"Error processing image {img_index} on page {i}: {e}")
            continue

doc.close()

# ======================================
# Build FAISS vector store
# ======================================
embeddings_array = np.array(all_embeddings)
vector_store = FAISS.from_embeddings(
    text_embeddings=[(doc.page_content, emb) for doc, emb in zip(all_docs, embeddings_array)],
    embedding=None,
    metadatas=[doc.metadata for doc in all_docs]
)

# ======================================
# Initialize Ollama LLM
# ======================================
llm = ChatOllama(
    model="llama3.1",  
    temperature=0.1
)

# ======================================
# Retrieve function with similarity threshold
# ======================================
def retrieve_multimodal(query, k=3, threshold=0.75):
    query_embedding = embed_text(query).reshape(1, -1)

    # Search in FAISS
    results = vector_store.similarity_search_by_vector(
        embedding=query_embedding.flatten(),
        k=k
    )

    filtered_results = []
    highest_score = 0.0

    for doc in results:
        # Use precomputed embedding from vector_store instead of all_docs.index()
        if doc.metadata["type"] == "text":
            doc_embedding = embed_text(doc.page_content).reshape(1, -1)
        else:
            img_id = doc.metadata["image_id"]
            doc_embedding = embed_image(
                Image.open(io.BytesIO(base64.b64decode(image_data_store[img_id])))
            ).reshape(1, -1)

        score = cosine_similarity(query_embedding, doc_embedding)[0][0]
        if score > highest_score:
            highest_score = score
        if score >= threshold:
            filtered_results.append(doc)

        print(f"Doc page {doc.metadata['page']} | type {doc.metadata['type']} | similarity {score:.3f}")

    return filtered_results, highest_score


# ======================================
# Create message for LLM
# ======================================
def create_multimodal_message(query, retrieved_docs):
    content = []
    context_text = f"Question: {query}\n\nContext from PDF:\n"
    for doc in retrieved_docs:
        if doc.metadata["type"] == "text":
            context_text += doc.page_content + "\n"
        elif doc.metadata["type"] == "image":
            context_text += f"[Image reference from page {doc.metadata['page']}]\n"

    context_text += "\nAnswer strictly based on the provided PDF content only. "\
                    "If answer is not present, reply 'Not in PDF'."
    content.append({"type": "text", "text": context_text})
    return HumanMessage(content=content)

# ======================================
# Main pipeline
# ======================================
def multimodal_pdf_rag_pipeline(query):
    context_docs, highest_score = retrieve_multimodal(query, k=3, threshold=0.75)

    if not context_docs or highest_score < 0.75:
        return "The information is not available in the PDF."

    message = create_multimodal_message(query, context_docs)
    response = llm.invoke([message])

    print(f"\nRetrieved {len(context_docs)} documents (highest similarity: {highest_score:.2f}):")
    for doc in context_docs:
        print(f"  - {doc.metadata['type']} from page {doc.metadata['page']}")
    print("\n")

    return response.content


# ======================================
# CLI testing
# ======================================
if __name__ == "__main__":
    queries = [
        "Tell me about Wildfire Damages",
        "Explain Percentage Acreage Burned by Ownership on Figure 3"
    ]
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        answer = multimodal_pdf_rag_pipeline(query)
        print(f"Answer: {answer}")
        print("=" * 70)
