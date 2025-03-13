import os
import json
import logging
from tqdm import tqdm
from config.logging_config import setup_logging  
from langchain_openai import OpenAIEmbeddings
from config.config import OPENAI_API_KEY

# Initialize OpenAI embeddings
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=OPENAI_API_KEY, dimensions=1024)

def calculate_embeddings(data):
    # Calculate embeddings for relevant fields based on the mapping
    fields_to_embed = [
        'contact_information_address',
        'personal_summary',
        'education_degrees',
        'education_field_of_study',
        'education_institutions',
        'education_locations',
        'education_honors',
        'education_descriptions',
        'work_experience_job_titles',
        'work_experience_employers',
        'work_experience_industry',
        'work_experience_employment_type',
        'work_experience_locations',
        'work_experience_descriptions',
        'skills',
        'project_titles',
        'project_project_types',
        'project_descriptions',
        'project_roles',
        'project_tools_technologies',
        'certification_names',
        'certification_descriptions',
        'certification_issuing_organizations',
        'publication_titles',
        'publication_publishers',
        'publication_descriptions',
        'language_languages',
        'language_proficiencies',
        'award_and_honor_titles',
        'award_and_honor_issuing_organizations',
        'award_and_honor_descriptions',
        'volunteer_experience_roles',
        'volunteer_experience_organizations',
        'volunteer_experience_descriptions'
    ]
    
    for field in fields_to_embed:
        if data.get(field) is not None and isinstance(data[field], str):
            # Add the embedding directly to the object
            embedding = embeddings_model.embed_query(data[field])
            data[f"{field}_embedding"] = embedding
    
    return data

def embed_json_file(input_file_path, output_file_path):
    setup_logging('embeddings_calculation')
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    # Skip processing if the output file already exists
    if os.path.exists(output_file_path):
        logging.info(f"Skipping {os.path.basename(input_file_path)} - already processed")
        return
    
    with open(input_file_path, 'r') as f:
        data = json.load(f)
    
    logging.info(f"Calculating embeddings for {os.path.basename(input_file_path)}")
    # Calculate embeddings
    updated_data = calculate_embeddings(data)
    
    # Save updated JSON with embeddings
    with open(output_file_path, 'w') as f:
        json.dump(updated_data, f, indent=2)
    logging.info(f"Successfully saved {os.path.basename(output_file_path)}")

def embed_json_files(input_dir, output_dir):
    setup_logging('embeddings_calculation')
    os.makedirs(output_dir, exist_ok=True)
    
    file_count = 0  # Initialize counter
    
    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith('.json'):
            input_file_path = os.path.join(input_dir, filename)
            output_file_path = os.path.join(output_dir, filename)
            
            # Skip processing if the output file already exists
            if os.path.exists(output_file_path):
                logging.info(f"Skipping {filename} - already processed")
                continue
            
            with open(input_file_path, 'r') as f:
                data = json.load(f)
            
            logging.info(f"Calculating embeddings for {filename}")
            # Calculate embeddings
            updated_data = calculate_embeddings(data)
            
            # Save updated JSON with embeddings
            with open(output_file_path, 'w') as f:
                json.dump(updated_data, f, indent=2)
            logging.info(f"Successfully saved {filename}")
            