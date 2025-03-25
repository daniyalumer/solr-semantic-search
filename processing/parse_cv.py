import os
import json
import logging
import pandas as pd
import pprint
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from config.config import OPENAI_API_KEY
from config.logging_config import setup_logging 
from processing.models_cv import ResponseFormatter

def create_extraction_prompt_cv(cv_text: str) -> ChatPromptTemplate:
    system_message = SystemMessage(
        content="""
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
    )

    user_message = HumanMessage(
        content=f"""
        Extract structured data from the CV text below, strictly following the schema rules.
        Parse the data into their appropriate fields based on the context and don’t place it in another field.
        Infer the industry from the work experience if not specified.

        **CV TEXT:**
        {cv_text}
        """
    )

    return ChatPromptTemplate.from_messages([system_message, user_message])


def parse_cv(raw_text: str, filename: str) -> Optional[ResponseFormatter]:
    """Parse CV text using LangChain and return structured data"""
    try:
        logging.info(f"Starting parsing for {filename}")
        prompt = create_extraction_prompt_cv(raw_text)

        prompt = prompt.format_messages()
        
        # Call LangChain API using invoke method
        logging.debug(f"Sending request to LangChain for {filename}")
        model = ChatOpenAI(
            temperature=0,
            model="gpt-4o",
            openai_api_key=OPENAI_API_KEY
        )

        model_with_structure = model.with_structured_output(ResponseFormatter)

        response = model_with_structure.invoke(prompt)

        pprint.pprint(response)
        
        # Extract JSON from response
        try:
            json_str = response.json()
            parsed_data = json.loads(json_str)  # Parse the JSON string
            
            # Add document identifier without .pdf extension
            document_id = os.path.splitext(filename)[0]  # Strip .pdf from filename
            parsed_data['document_id'] = document_id  # Add the document identifier
            
            print(parsed_data)
            logging.info(f"Successfully parsed and validated {filename}")
            return parsed_data
            
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error for {filename}: {e}")
            return None
            
    except Exception as e:
        logging.error(f"Error during CV parsing for {filename}: {e}")
        return None
    
def process_cvs(csv_path: str, output_dir: str, batch_size: int = 10):
    """Process all CVs from CSV and save parsed results"""
    try:
        # Setup logging
        setup_logging('cv_parsing')
        logging.info(f"Starting CV processing with batch size {batch_size}")
        
        # Read CSV with raw CV text
        df = pd.read_csv(csv_path)
        logging.info(f"Loaded {len(df)} CVs from {csv_path}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process CVs in batches
        successful_parses = 0
        failed_parses = 0
        skipped_files = 0

        # Process only one batch of CVs
        logging.info(f"Processing batch 1/1")
        batch = df.iloc[0:batch_size]  # Get the first batch
        
        #for i in range(0, len(df), batch_size):
        #    batch = df.iloc[i:i + batch_size]
        #    logging.info(f"Processing batch {i//batch_size + 1}/{(len(df) + batch_size - 1) // batch_size}")
            
        for _, row in batch.iterrows():
            output_file = os.path.join(output_dir, f"{os.path.splitext(row['filename'])[0]}.json")
            
            # Skip if already processed
            if os.path.exists(output_file):
                logging.info(f"Skipping {row['filename']} - already processed")
                skipped_files += 1
                continue
            
            # Parse CV
            parsed_data = parse_cv(row['preprocessed_text'], row['filename'])
            if parsed_data:
                # Save to JSON file
                try:
                    with open(output_file, 'w') as f:
                        json.dump(parsed_data, f, indent=2)
                    logging.info(f"Successfully saved {row['filename']}")
                    successful_parses += 1
                except Exception as e:
                    logging.error(f"Error saving {row['filename']}: {e}")
                    failed_parses += 1
            else:
                failed_parses += 1
        
        # Log summary statistics
        logging.info("Processing completed!")
        logging.info(f"Successfully parsed: {successful_parses}")
        logging.info(f"Failed to parse: {failed_parses}")
        logging.info(f"Skipped files: {skipped_files}")
        
    except Exception as e:
        logging.error(f"Error during batch processing: {e}")
        raise


