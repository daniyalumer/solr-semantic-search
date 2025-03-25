import os
import json
import pandas as pd
from datetime import datetime
import logging
import requests
from config.logging_config import setup_logging
from config.config import OPENAI_API_KEY
from typing import Optional

def create_extraction_prompt_cv_bench(cv_text: str) -> str:
    system_message = """
    You are a CV parsing assistant. Your task is to extract structured information from the provided CV. 
    Return **only** a valid JSON object that strictly follows the predefined schema—no additional text, comments, or formatting.

    **Strict Output Requirements:**
    - The response **must** be a valid JSON object adhering to the schema exactly.
    - **No extra text, explanations, or formatting** are allowed (e.g., no markdown, no "```json" wrappers).
    - All schema keys must be present; missing values should be **null**.
    - Use **exact key names** as defined—do not modify, add, or remove keys.
    - Ensure **correct data types** (`boolean`, `float`, `string`, etc.) as specified in the schema.
    - Dates must follow **`YYYY-MM-DD`** format (e.g., `"2025-02-13"`).
    - If a date is `"present"`, `"current"`, or `"now"`, return `"9999-12-31"`.
    - Infer missing fields like `industry` from context (e.g., work experience) when possible.
    - Use enum values (e.g., `EmploymentType`, `ProficiencyLevel`) exactly as defined—do not invent new ones.

    **Non-Compliance Warning:**
    - Any deviation from the schema (e.g., missing keys, wrong data types, extraneous text) will be considered an incorrect response.

    **Schema:**
    {
        "contact_information_full_name": str | null,
        "contact_information_phone_number": str | null,
        "contact_information_email": str | null,
        "contact_information_linkedin": str | null,
        "contact_information_address": str | null,
        "personal_summary": str | null,
        "education_degrees": str | null,
        "education_field_of_study": str | null,
        "education_institutions": str | null,
        "education_locations": str | null,
        "education_honors": str | null,
        "education_descriptions": str | null,
        "work_experience_job_titles": str | null,
        "work_experience_employers": str | null,
        "work_experience_industry": str | null,
        "work_experience_employment_type": "full_time" | "part_time" | "contract" | "freelance" | "internship" | "volunteer" | null,
        "work_experience_locations": str | null,
        "work_experience_descriptions": str | null,
        "work_experience_seniority": "junior" | "mid" | "senior" | null,
        "skills": str | null,
        "project_titles": str | null,
        "project_project_types": str | null,
        "project_descriptions": str | null,
        "project_roles": str | null,
        "project_tools_technologies": str | null,
        "certification_names": str | null,
        "certification_descriptions": str | null,
        "certification_issuing_organizations": str | null,
        "publication_titles": str | null,
        "publication_publishers": str | null,
        "publication_descriptions": str | null,
        "language_languages": str | null,
        "language_proficiencies": str | null,
        "award_and_honor_titles": str | null,
        "award_and_honor_issuing_organizations": str | null,
        "award_and_honor_descriptions": str | null,
        "volunteer_experience_roles": str | null,
        "volunteer_experience_organizations": str | null,
        "volunteer_experience_descriptions": str | null
    }

    **Additional Instructions:**
    - Parse the CV contextually to assign data to the correct fields (e.g., don’t misplace a job title as a project title).
    - Calculate `total_work_experience` as the sum of all work experience durations in years (approximate if exact dates are missing).
    - For `skills`, concatenate all mentioned skills into a single string, separated by commas.
    - Use today’s date (2025-02-27) as a reference for "present" calculations if needed.
    - Infer `work_experience_seniority` based on the total work experience in years:
      - Less than 3 years: "junior"
      - 3 to 6 years: "mid"
      - More than 6 years: "senior"
    """

    user_message = f"""
    Extract structured data from the CV text below, strictly following the schema rules.
    Parse the data into their appropriate fields based on the context and don’t place it in another field.
    Infer the industry from the work experience if not specified.

    **CV TEXT:**
    {cv_text}
    """

    return system_message + "\n\n" + user_message

def call_openai_api(prompt: str, model: str, temperature: float = 0, endpoint: str = "completions") -> dict:
    """Call OpenAI API and return the response."""
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    if endpoint == "completions":
        url = "https://api.openai.com/v1/chat/completions"
        data = {
            "model": model,
            "messages": [{"role": "system", "content": prompt}],
            "temperature": temperature
        }
    elif endpoint == "embeddings":
        url = "https://api.openai.com/v1/embeddings"
        data = {
            "model": model,
            "input": prompt
        }
    else:
        raise ValueError("Invalid endpoint specified")

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()

def parse_cv(raw_text: str, filename: str) -> Optional[dict]:
    """Parse CV text using OpenAI API and return structured data."""
    try:
        logging.info(f"Starting parsing for {filename}")
        prompt = create_extraction_prompt_cv_bench(raw_text)
        
        logging.debug(f"Sending request to OpenAI for {filename}")
        response = call_openai_api(prompt, model="gpt-4o", endpoint="completions")
        
        logging.info(f"OpenAI response for {filename}: {response}")
        
        # Extract JSON from response
        try:
            parsed_data = response['choices'][0]['message']['content']  # Adjust based on actual response structure
            parsed_data = json.loads(parsed_data)  # Parse the JSON string
            
            # Add document identifier without .pdf extension
            document_id = os.path.splitext(filename)[0]  # Strip .pdf from filename
            parsed_data['document_id'] = document_id  # Add the document identifier
            
            print(parsed_data)
            logging.info(f"Successfully parsed and validated {filename}")
            return response, parsed_data
            
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error for {filename}: {e}")
            return None
            
    except Exception as e:
        logging.error(f"Error during CV parsing for {filename}: {e}")
        return None

def calculate_embeddings(parsed_data: dict) -> Optional[dict]:
    """Calculate embeddings using OpenAI API and return the response."""
    try:
        logging.info(f"Starting embeddings calculation for {parsed_data['document_id']}")
        prompt = json.dumps(parsed_data)  # Convert parsed data to JSON string
        
        logging.debug(f"Sending request to OpenAI for embeddings for {parsed_data['document_id']}")
        response = call_openai_api(prompt, model="text-embedding-ada-002", endpoint="embeddings")  # Use the appropriate model for embeddings
        
        logging.info(f"OpenAI response for embeddings for {parsed_data['document_id']}: {response}")
        
        return response
        
    except Exception as e:
        logging.error(f"Error during embeddings calculation for {parsed_data['document_id']}: {e}")
        return None

def record_parsing_details(cv_id, request_body, response_body, start_time, end_time, elapsed_time, log_file, operation):
    """Record the parsing details to the terminal."""
    log_entry = {
        "cv_id": cv_id,
        "request_body": request_body,
        "start_time": start_time,
        "end_time": end_time,
        "elapsed_time": elapsed_time,
        "operation": operation
    }
    if operation == "parsing":
        log_entry["response_body"] = response_body
    elif operation == "embedding" and response_body:
        # Remove the vector from the response body
        if 'data' in response_body and len(response_body['data']) > 0:
            response_body['data'][0].pop('embedding', None)
        log_entry["response_body"] = response_body
    print(json.dumps(log_entry, indent=2))

def parse_and_embed_cvs(input_csv, output_dir, log_file):
    """Parse CVs and calculate embeddings, recording details."""
    setup_logging('parse_and_embed_cvs')
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(input_csv)
    count = 0
    for _, row in df.iterrows():
        if count >= 5:
            break
        cv_id = row['filename']
        request_body = row['preprocessed_text']
        
        # Timing for parsing
        parse_start_time = datetime.now()
        parsed_data = parse_cv(request_body, cv_id)
        parse_end_time = datetime.now()
        parse_elapsed_time = (parse_end_time - parse_start_time).total_seconds()
        
        if parsed_data:
            response_body = json.dumps(parsed_data)
            record_parsing_details(cv_id, request_body, response_body, parse_start_time.isoformat(), parse_end_time.isoformat(), parse_elapsed_time, log_file, "parsing")
            
            # Timing for embeddings
            embed_start_time = datetime.now()
            embeddings_data = calculate_embeddings(parsed_data[1])  # Pass the parsed data for embeddings
            embed_end_time = datetime.now()
            embed_elapsed_time = (embed_end_time - embed_start_time).total_seconds()
            
            record_parsing_details(cv_id, request_body, embeddings_data, embed_start_time.isoformat(), embed_end_time.isoformat(), embed_elapsed_time, log_file, "embedding")
        else:
            print(f"Failed to parse CV: {cv_id}")
        
        count += 1

if __name__ == "__main__":
    input_csv = "data/extracted/cv.csv"
    output_dir = "data/parsed_data/cv"
    log_file = "logs/parsing_log.json"
    
    parse_and_embed_cvs(input_csv, output_dir, log_file)