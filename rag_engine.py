import ollama
from database import get_vector_collection

# --- CONFIGURATION ---
LLM_MODEL = "phi4-mini"
COLLECTION = get_vector_collection()

def get_relevant_context(query, n_results=6):

    results = COLLECTION.query(
        query_texts=[query],
        n_results=n_results
    )
    
    context_list = []
    # Loop through the retrieved documents and their metadata
    for i in range(len(results['documents'][0])):
        text = results['documents'][0][i]
        meta = results['metadatas'][0][i]
        
        # Format a clear reference for the LLM
        source_info = f"[Source: {meta['source']}, Page: {meta['page']}]"
        context_list.append(f"{source_info}\n{text}")
    
    return "\n\n".join(context_list)

def generate_answer(query):

    # 1. Get the facts from your database
    context = get_relevant_context(query)
    
    # 2. Construct a strict prompt
    prompt = f"""
    you are an scholarship assistant for NED University. you will get access to a certain context which will include information about different scholarships.
    your job is to analyse and understand all the scholarship information and then answer the user query. 
    Answer the question ONLY using the provided context. 
    If the context doesn't contain the answer, say "I don't have this specific information in my database."
    
    CONTEXT:
    {context}
    
    QUESTION: {query}
    
    ANSWER (including citations):"""

    # 3. Call Ollama
    print(f"Thinking using {LLM_MODEL}...")
    response = ollama.generate(model=LLM_MODEL, prompt=prompt)
    return response['response']

if __name__ == "__main__":
    user_query = "give me some information about the majee scholarship"
    print("\n--- AI RESPONSE ---")
    print(generate_answer(user_query))