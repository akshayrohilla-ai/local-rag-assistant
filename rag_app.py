from sentence_transformers import SentenceTransformer
import chromadb
from ollama import chat

# ----------------------------
# Load embedding model
# ----------------------------
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# ----------------------------
# Read document
# ----------------------------
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Simple manual chunks
chunks = [
    text[0:200],
    text[200:400],
    text[400:600]
]

# ----------------------------
# Create vector DB
# ----------------------------
client = chromadb.Client()
collection = client.create_collection("knowledge_base")

# Add chunks with embeddings
for i, chunk in enumerate(chunks):
    embedding = embed_model.encode(chunk).tolist()

    collection.add(
        ids=[str(i)],
        embeddings=[embedding],
        documents=[chunk]
    )

print("RAG Assistant Ready (type exit to quit)\n")

# ----------------------------
# Chat Loop
# ----------------------------
while True:

    query = input("You: ")

    if query.lower() == "exit":
        break

    # Embed user question
    query_embedding = embed_model.encode(query).tolist()

    # Retrieve top matching chunk
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )

    context = "\n".join(results["documents"][0])
# If retrieval looks weak, refuse answer
    if len(context.strip()) < 50:
        print("\nAssistant:")
        print("I don't know.")
        print("\n"+"-"*50+"\n")
        continue
    # Ground model with retrieved context
    response = chat(
        model='phi3:mini',
        messages=[
            {
                'role': 'system',
                'content':
                '''
Answer ONLY using the supplied context.

Keep answers concise (maximum 3 lines).

Do not add outside knowledge.
Do not elaborate beyond context.

If not in context say:
I don't know.
'''
            },
            {
                'role':'user',
                'content': f"Context:\n{context}\n\nQuestion:\n{query}"
            }
        ]
    )

    answer = response['message']['content']
    # Guardrail for concise RAG answers
    lines = answer.split('\n')
    answer = '\n'.join(lines[:3])

    print("\nAssistant:")
    print(answer)
    print("\n"+"-"*50+"\n")