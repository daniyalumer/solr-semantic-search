import time
import pandas as pd
import os
import numpy as np
from langchain_openai import OpenAIEmbeddings
import logging
from config.config import OPENAI_API_KEY
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('embeddings.log')
    ]
)

# Initialize the OpenAI embeddings model
print("Setting up OpenAI embeddings...")
start_time = time.time()

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
    if not isinstance(text, str) or not text.strip():
        return np.zeros(1024).tolist()  # Return zero vector if text is empty or invalid
    try:
        embedding = embeddings_model.embed_query(text)
        return embedding
    except Exception as e:
        logging.error(f"Error generating embedding for text: {text[:30]}... - {e}")
        return np.zeros(1024).tolist()  # Return zero vector in case of error

def calculate_embeddings(input_csv: str, output_csv: str):
    """
    Reads a CSV file, generates embeddings for specific fields, and saves the updated CSV.
    """
    try:
        if not os.path.exists(input_csv):
            logging.error(f"Input CSV file '{input_csv}' not found.")
            return

        logging.info(f"Loading CSV file: {input_csv}")
        df = pd.read_csv(input_csv)
        logging.info(f"Loaded CSV with {len(df)} rows.")

        # Generate embeddings for each field
        logging.info("Generating embeddings for job data...")
        df["title_vector"] = df["Job Title"].apply(embed_text_openai)
        logging.info("Job titles embedded.")
        df["desc_vector"] = df["Job Description"].apply(embed_text_openai)
        logging.info("Descriptions embedded.")
        df["location_vector"] = df["Location"].apply(embed_text_openai)
        logging.info("Locations embedded.")
        df["skills_vector"] = df["Required Skills"].apply(embed_text_openai)
        logging.info("Skills embedded.")

        # Save the updated CSV with embeddings
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        logging.info(f"Embeddings saved to {output_csv}")
    except Exception as e:
        logging.error(f"Error in calculate_embeddings: {e}", exc_info=True)

if __name__ == "__main__":
    # Define input and output CSV file paths
    input_csv = "/Users/danya1/Desktop/solr-semantic-search/data/extracted_jd/rozee_jobs_llm.csv"
    output_csv = "/Users/danya1/Desktop/solr-semantic-search/data/extracted_jd/rozee_jobs_with_embeddings.csv"

    # Call the calculate_embeddings function
    calculate_embeddings(input_csv, output_csv)