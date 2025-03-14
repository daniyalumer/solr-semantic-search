import json
import csv
import os
import logging
import traceback
import re
from openai import OpenAI, AuthenticationError, RateLimitError, APIError
from config.config import OPENAI_API_KEY
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('processor.log')
    ]
)

# Initialize OpenAI client
try:
    logging.info("Initializing OpenAI client")
    client = OpenAI(api_key=" OPENAI_API_KEY ")

except Exception as e:
    logging.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
    raise

def is_url_in_csv(url):
    """Check if a URL already exists in the CSV file."""
    csv_file = "/Users/danya1/Desktop/solr-semantic-search/data/extracted_jd/rozee_jobs_llm.csv"
    try:
        if not os.path.exists(csv_file):
            logging.debug(f"CSV file {csv_file} does not exist yet")
            return False
        with open(csv_file, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row["Link"] == url:
                    logging.debug(f"Found existing URL in CSV: {url}")
                    return True
        return False
    except Exception as e:
        logging.error(f"Error checking URL in CSV: {e}", exc_info=True)
        return False

def determine_seniority(experience):
    """Determine seniority based on years of experience."""
    if not experience or experience == "NULL":
        return "NULL"
    try:
        exp_str = experience.lower().replace("+", "").strip()
        if "-" in exp_str:
            parts = exp_str.split("-")
            low = float(''.join(c for c in parts[0] if c.isdigit() or c == '.'))
            high = float(''.join(c for c in parts[1] if c.isdigit() or c == '.'))
            years = (low + high) / 2
        else:
            years = float(''.join(c for c in exp_str if c.isdigit() or c == '.'))
        
        if years < 3:
            return "Junior"
        elif 3 <= years <= 5:
            return "Mid"
        else:
            return "Senior"
    except (ValueError, IndexError) as e:
        logging.warning(f"Could not parse experience: {experience}, error: {e}, defaulting to NULL")
        return "NULL"

def extract_json_from_text(text):
    """Extract JSON from text that might contain markdown, code blocks, or other text."""
    json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    matches = re.findall(json_pattern, text)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    json_pattern = r'\{[\s\S]*\}'
    matches = re.findall(json_pattern, text)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    return None

def parse_text_with_llm(job_text):
    """Parse job text using OpenAI to extract structured data."""
    sanitized_job_text = job_text.replace("{", "{{").replace("}", "}}")
    prompt = """Extract the following information from the job posting text and return it as a valid JSON object. 
Ensure the response is proper JSON format with all fields. If any field is missing, use "NULL":
- Job Title
- Company Name
- Location
- Job Description
- Required Skills
- Experience (in years, e.g., "2 years", "5+ years", "2-3 years", or "NULL" if not specified)

Format your response as a valid JSON object:
{
  "Job Title": "Title goes here",
  "Company Name": "Company goes here",
  "Location": "Location goes here",
  "Job Description": "Description goes here",
  "Required Skills": "Skills go here",
  "Experience": "Experience goes here"
}

Job Posting Text:
"""
    prompt += sanitized_job_text[:4000]

    try:
        logging.info("Sending request to OpenAI API")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts structured data from text. You always return valid JSON without any additional text or explanations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.3
        )
        llm_output = response.choices[0].message.content.strip()
        if not llm_output:
            logging.error("OpenAI returned an empty response")
            return get_default_job_data(job_text)

        structured_data = extract_json_from_text(llm_output)
        if not structured_data:
            logging.error(f"Could not extract JSON from OpenAI response: '{llm_output[:200]}...'")
            return get_default_job_data(job_text)

        required_fields = ["Job Title", "Company Name", "Location", "Job Description", "Required Skills", "Experience"]
        for field in required_fields:
            if field not in structured_data:
                structured_data[field] = "NULL"

        experience = structured_data.get("Experience", "NULL")
        structured_data["Seniority"] = determine_seniority(experience)
        return structured_data
    except (AuthenticationError, RateLimitError, APIError, json.JSONDecodeError) as e:
        logging.error(f"Error during OpenAI processing: {e}", exc_info=True)
        return get_default_job_data(job_text)
    except Exception as e:
        logging.error(f"Unexpected error in OpenAI processing: {e}", exc_info=True)
        return get_default_job_data(job_text)

def get_default_job_data(job_text):
    """Return default job data when parsing fails."""
    return {
        "Job Title": "NULL",
        "Company Name": "NULL",
        "Location": "NULL",
        "Job Description": job_text[:100] if job_text else "NULL",
        "Required Skills": "NULL",
        "Experience": "NULL",
        "Seniority": "NULL"
    }

def save_to_csv(job_data):
    """Save job data to a CSV file."""
    csv_file = "/Users/danya1/Desktop/solr-semantic-search/data/extracted_jd/rozee_jobs_llm.csv"
    try:
        # Ensure the directory exists
        directory = os.path.dirname(csv_file)
        if not os.path.exists(directory):
            logging.info(f"Directory does not exist. Creating: {directory}")
            os.makedirs(directory, exist_ok=True)

        # Check if the file already exists
        file_exists = os.path.exists(csv_file) and os.stat(csv_file).st_size > 0

        # Open the file in append mode and write data
        with open(csv_file, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if not file_exists:
                # Write the header row if the file is new
                writer.writerow(["Job Title", "Company Name", "Location", "Job Description", "Required Skills", "Experience", "Seniority", "Link"])
            # Write the job data
            writer.writerow([
                job_data.get("Job Title", "NULL"),
                job_data.get("Company Name", "NULL"),
                job_data.get("Location", "NULL"),
                job_data.get("Job Description", "NULL"),
                job_data.get("Required Skills", "NULL"),
                job_data.get("Experience", "NULL"),
                job_data.get("Seniority", "NULL"),
                job_data.get("url", "NULL")
            ])
        logging.info(f"Appended job data for {job_data.get('url', 'unknown')} to CSV")
    except Exception as e:
        logging.error(f"Error writing to CSV: {e}", exc_info=True)
        raise

def process_job_function(job_text, job_url):
    """Process a job posting by extracting structured data and saving it to a CSV file."""
    try:
        logging.info("Processing job via function call")
        if not job_text or not job_url:
            missing = []
            if not job_text:
                missing.append("job_text")
            if not job_url:
                missing.append("job_url")
            logging.warning(f"Missing required fields: {', '.join(missing)}")
            return {"error": f"Missing required fields: {', '.join(missing)}"}

        if is_url_in_csv(job_url):
            logging.info(f"Job already processed: {job_url}")
            return {"message": "Job already processed"}

        structured_data = parse_text_with_llm(job_text)
        structured_data['url'] = job_url
        save_to_csv(structured_data)
        logging.info(f"Successfully processed job: {job_url}")
        return {"message": "Job processed successfully", "data": structured_data}

    except Exception as e:
        logging.error(f"Error in process_job_function: {e}", exc_info=True)
        traceback_str = traceback.format_exc()
        logging.debug(f"Full traceback: {traceback_str}")
        return {"error": "Internal server error", "details": str(e)}