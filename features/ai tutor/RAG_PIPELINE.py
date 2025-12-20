#import and install
# !pip install pypdf chromadb sentence-transformers --quiet

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import pipeline


#step 1---LOAD PDF
def load_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"

    return text

#step 2 ---CHUNK THE DATA
def chunk_text(text, chunk_size=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks


#step 3--- EMBEDDING MODEL
embedder = SentenceTransformer("all-MiniLM-L6-v2")

#STEP 4---CREATE VECTOR DATABASE
client = chromadb.Client()
collection = client.get_or_create_collection(
    name="rag_collection",
    metadata={"hnsw:space": "cosine"}) #required in new api


# STEP 5---clear previous collection
def clear_collection():
    data = collection.get()
    ids = data.get("ids", [])
    if ids:   # only delete if not empty
        collection.delete(ids=ids)
        print(f"Deleted {len(ids)} old vectors.")
    else:
        print("Collection already empty.")



#STEP 6---Add chunks to vector DB
def add_to_chroma(chunks):
    ids = [str(i) for i in range(len(chunks))]
    embeddings = embedder.encode(chunks).tolist()

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks
    )

# STEP 7---Retrieve top-k chunks
def retrieve(query, k=3):
    query_emb = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_emb,
        n_results=k
    )
    return results["documents"][0][0]


# STEP 8---Simple Question Generator
def generate_questions(context, n=5):
    sentences = context.split(".")
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    questions = []
    for s in sentences[:n]:
        q = f"What is {s[:70]}?"
        questions.append(q)

    return questions


# **MAIN PIPELINE**


# MAIN PIPELINE
pdf_path = "/content/Kritika_Work_Report.pdf"  # <-- Upload your PDF and replace path


print("\nLoading PDF...")
text = load_pdf(pdf_path)

print("Chunking text...")
chunks = chunk_text(text)

print("Clearing old data from Chroma...")
clear_collection()

print("Adding new chunks to Chroma...")
add_to_chroma(chunks)

print("Retrieving top context...")
retrieved = retrieve("important concepts")

print("\nRetrieved Context:")
print(retrieved)

print("\nGenerated Questions:")
for q in generate_questions(retrieved):
    print("-", q)


# **ADDING RERANKING LAYER FOR BETTER PERFORMANCE**


# !pip install rank_bm25


from rank_bm25 import BM25Okapi

def rerank_bm25(query, chunks):
    tokenized_chunks = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    scores = bm25.get_scores(query.lower().split())

    # sort chunks by score (descending)
    ranked = [x for _, x in sorted(zip(scores, chunks), reverse=True)]
    return ranked[:3]   # top 3 chunks after reranking


def retrieve_with_rerank(topic, top_k=5):
    # step 1: retrieve from Chroma
    results = collection.query(query_texts=[topic], n_results=top_k)
    retrieved_chunks = results["documents"][0]

    # step 2: bm25 rerank
    reranked = rerank_bm25(topic, retrieved_chunks)

    return reranked


def generate_quiz_with_context(topic, difficulty):
    retrieved_chunks = retrieve_with_rerank(topic)

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
    You are a quiz generator.

    Topic: {topic}
    Difficulty: {difficulty}

    Use this context to create 5 high-quality MCQs:
    {context}

    Output format:
    Q1)
    A)
    B)
    C)
    D)
    Answer:
    """

    return prompt


for t in ["Math", "Biology", "History"]:
    print("---------")
    print(generate_quiz_with_context(t, "easy"))
    print("---------")


#pipeline
def run_rag_pipeline(pdf_path, topic="General", difficulty="easy"):
    print("\n--- RUNNING RAG PIPELINE ---")

    # 1. Load PDF
    print("Loading PDF...")
    text = load_pdf(pdf_path)

    # 2. Chunk
    print("Chunking text...")
    chunks = chunk_text(text)

    # 3. Reset Vector DB
    print("Clearing old Chroma collection...")
    clear_collection()

    # 4. Add to Chroma
    print("Adding chunks to ChromaDB...")
    add_to_chroma(chunks)

    # 5. Retrieve from Chroma normally
    print("Retrieving top chunks from Chroma...")
    results = collection.query(query_texts=[topic], n_results=5)
    retrieved_chunks = results["documents"][0]

    # 6. BM25 RERANK CALL
    print("Applying BM25 reranking...")
    top_chunks = rerank_bm25(topic, retrieved_chunks)[:3]

    # 7. Context join
    context = "\n\n".join(top_chunks)

    # 8. Generate quiz using this context
    print("Generating Quiz...\n")
    quiz = generate_quiz_with_context(topic, difficulty)

    print("----- QUIZ GENERATED -----")
    print(quiz)

    return quiz


#test for history quiz
run_rag_pipeline("/content/history_sample.pdf", topic="History", difficulty="easy")


#test for biology quiz
run_rag_pipeline("/content/biology_sample.pdf", topic="History", difficulty="easy")


#test for maths quiz
run_rag_pipeline("/content/maths_simple.pdf", topic="Maths", difficulty="hard")