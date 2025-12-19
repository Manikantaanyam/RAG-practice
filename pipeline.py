from dotenv import load_dotenv
from retireiver.retrieve import rag_retreiver
from openai import OpenAI
import os
import json

load_dotenv()
key = os.getenv("GEMINI_API_KEY") 


def rag_simple(query, retriever, client, top_k=3):
    results = retriever.retrieve(query, top_k=top_k)
    
    context = "\n\n".join([doc['content'] for doc in results]) if results else ""
    if not context:
        return "No relevant context found"
    
    prompt = f"""You are an expert at explaining things. Use the following context to answer clearly.

Context:
{context}

Question:
{query}

Answer in JSON format.
"""
    
    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ]
    )
    
    # Print model response content
    print("answert ", json.dumps(response.choices[0].message.model_dump(), indent=2))


client = OpenAI(
    api_key=key,
    # This base_url points to the Gemini API via the OpenAI compatibility layer
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

rag_simple("""Considering a Long List of Machine Learning Algorithms, given a Data Set, How Do You 
Decide Which One to Use""", rag_retreiver, client, top_k=3)
