import time
import pandas as pd
import os
import numpy as np
from langchain_openai import OpenAIEmbeddings
from config.config import OPENAI_API_KEY


print("Setting up OpenAI embeddings...")
start_time = time.time()

# Initialize the OpenAI embeddings model
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-large", 
    openai_api_key="OPENAI_API_KEY",
    dimensions=1024
)

end_time = time.time()
print(f"Setup completed in {end_time - start_time:.2f} seconds.")

def embed_text_openai(text) -> list:
    """
    Generates embeddings for the given text using OpenAI's embedding model.
    If the text is empty or not a string, returns a zero vector.
    Uses the text-embedding-3-large model with 1024 dimensions.
    """
    # Ensure text is a string; if it's NaN, set to empty string.
    if not isinstance(text, str):
        # Check for NaN (using pd.isna)
        if pd.isna(text):
            text = ""
        else:
            text = str(text)
    
    if not text or not text.strip():
        # Return zero vector with 1024 dimensions as a proper list
        return np.zeros(1024).tolist()
    
    try:
        # Use the embed_query method to get the embedding
        embedding = embeddings_model.embed_query(text)
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        # Return zeros in case of error as a proper list
        return np.zeros(1024).tolist()


# Read the CSV file containing the job details.
df = pd.read_csv("rozee_jobs_llm.csv")
print("Loaded CSV with", len(df), "rows.")

# Generate embeddings for each field.
print("Generating embeddings with OpenAI API - this may take some time...")
df["title_vector"] = df["Job Title"].apply(embed_text_openai)
print("Job titles embedded.")
df["desc_vector"] = df["Job Description"].apply(embed_text_openai)
print("Descriptions embedded.")
df["location_vector"] = df["Location"].apply(embed_text_openai)
print("Locations embedded.")
df["skills_vector"] = df["Required Skills"].apply(embed_text_openai)
print("Skills embedded.")

# Save the new CSV file with embedding columns
df.to_csv("data/extracted_jd/rozee_jobs_with_embeddings.csv", index=False)

print("✅ OpenAI embeddings generated and saved to rozee_jobs_with_embeddings_openai.csv")

