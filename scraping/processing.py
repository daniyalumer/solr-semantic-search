from flask import Flask, request, jsonify
import json
import csv
import os
from openai import OpenAI, AuthenticationError, RateLimitError, APIError
import logging
import traceback
from config.config import OPENAI_API_KEY
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('processor.log')
    ]
)

# Configuration
OPENAI_API_KEY = "OPENAI_API"
try:
    logging.info("Initializing Flask application")
    app = Flask(__name__)
except Exception as e:
    logging.error(f"Failed to initialize Flask: {e}", exc_info=True)
    raise

try:
    logging.info("Initializing OpenAI client")
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    logging.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
    raise

def is_url_in_csv(url):
    csv_file = "rozee_jobs_llm.csv"
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
    if not experience or experience == "N/A":
        return "N/A"
    try:
        exp_str = experience.lower().replace("+", "").strip()
        if "-" in exp_str:
            parts = exp_str.split("-")
            # Make sure we're extracting numeric values
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
        logging.warning(f"Could not parse experience: {experience}, error: {e}, defaulting to N/A")
        return "N/A"

def extract_json_from_text(text):
    """Extract JSON from text that might contain markdown, code blocks, or other text."""
    # Try to find JSON in code blocks
    json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    matches = re.findall(json_pattern, text)
    
    # If found in code blocks, try each match
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    # If no valid JSON in code blocks, try the whole text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to find anything that looks like JSON (between curly braces)
    json_pattern = r'\{[\s\S]*\}'
    matches = re.findall(json_pattern, text)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    # If all else fails, return None
    return None

def parse_text_with_llm(job_text):
    # First, sanitize the job text to avoid string formatting issues
    # Replace any curly braces that might be interpreted as format placeholders
    sanitized_job_text = job_text.replace("{", "{{").replace("}", "}}")
    
    prompt = """Extract the following information from the job posting text and return it as a valid JSON object. 
Ensure the response is proper JSON format with all fields. If any field is missing, use "N/A":
- Job Title
- Company Name
- Location
- Job Description
- Required Skills
- Experience (in years, e.g., "2 years", "5+ years", "2-3 years", or "N/A" if not specified)

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
    # Add the job text separately to avoid string formatting issues
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
            temperature=0.3  # Lower temperature for more consistent formatting
        )
        llm_output = response.choices[0].message.content.strip()
        logging.debug(f"Raw OpenAI response length: {len(llm_output)}")
        logging.debug(f"Raw OpenAI response first 100 chars: '{llm_output[:100]}'")

        if not llm_output:
            logging.error("OpenAI returned an empty response")
            return get_default_job_data(job_text)

        # Try to extract JSON from the response
        structured_data = extract_json_from_text(llm_output)
        
        if not structured_data:
            logging.error(f"Could not extract JSON from OpenAI response: '{llm_output[:200]}...'")
            return get_default_job_data(job_text)
            
        logging.info("Successfully parsed OpenAI response")

        # Ensure all required fields exist
        required_fields = ["Job Title", "Company Name", "Location", "Job Description", "Required Skills", "Experience"]
        for field in required_fields:
            if field not in structured_data:
                structured_data[field] = "N/A"

        # Add seniority based on experience
        experience = structured_data.get("Experience", "N/A")
        structured_data["Seniority"] = determine_seniority(experience)
        return structured_data
    except AuthenticationError as e:
        logging.error(f"OpenAI authentication error: {e}", exc_info=True)
        return get_default_job_data(job_text)
    except RateLimitError as e:
        logging.error(f"OpenAI rate limit exceeded: {e}", exc_info=True)
        return get_default_job_data(job_text)
    except APIError as e:
        logging.error(f"OpenAI API error: {e}", exc_info=True)
        logging.debug(f"Problematic job text: {job_text[:1000]}...")
        return get_default_job_data(job_text)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse OpenAI response as JSON: {e}", exc_info=True)
        logging.debug(f"Raw output causing JSON error: '{llm_output}'")
        return get_default_job_data(job_text)
    except Exception as e:
        logging.error(f"Unexpected error in OpenAI processing: {e}", exc_info=True)
        logging.debug(f"Problematic job text: {job_text[:1000]}...")
        return get_default_job_data(job_text)

def get_default_job_data(job_text):
    """Return default job data when parsing fails."""
    return {
        "Job Title": "N/A",
        "Company Name": "N/A",
        "Location": "N/A",
        "Job Description": job_text[:100] if job_text else "N/A",
        "Required Skills": "N/A",
        "Experience": "N/A",
        "Seniority": "N/A"
    }

@app.route('/process_job', methods=['POST'])
def process_job():
    try:
        logging.info("Received POST request to /process_job")
        if not request.is_json:
            logging.warning("Request content type is not JSON")
            return jsonify({"error": "Content-Type must be application/json"}), 415

        data = request.json
        job_text = data.get('job_text', '')
        job_url = data.get('job_url', '')

        if not job_text or not job_url:
            missing = []
            if not job_text:
                missing.append("job_text")
            if not job_url:
                missing.append("job_url")
            logging.warning(f"Missing required fields: {', '.join(missing)}")
            return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

        if is_url_in_csv(job_url):
            logging.info(f"Job already processed: {job_url}")
            return jsonify({"message": "Job already processed"}), 200

        structured_data = parse_text_with_llm(job_text)
        structured_data['url'] = job_url
        save_to_csv(structured_data)
        logging.info(f"Successfully processed job: {job_url}")
        return jsonify({"message": "Job processed successfully", "data": structured_data}), 200

    except Exception as e:
        logging.error(f"Error in process_job endpoint: {e}", exc_info=True)
        traceback_str = traceback.format_exc()
        logging.debug(f"Full traceback: {traceback_str}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

def save_to_csv(job_data):
    csv_file = "data/extracted_jd/rozee_jobs_llm.csv"
    try:
        file_exists = os.path.exists(csv_file) and os.stat(csv_file).st_size > 0
        with open(csv_file, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if not file_exists:
                logging.info(f"Creating new CSV file: {csv_file}")
                writer.writerow(["Job Title", "Company Name", "Location", "Job Description", "Required Skills", "Experience", "Seniority", "Link"])
            writer.writerow([
                job_data.get("Job Title", "N/A"),
                job_data.get("Company Name", "N/A"),
                job_data.get("Location", "N/A"),
                job_data.get("Job Description", "N/A"),
                job_data.get("Required Skills", "N/A"),
                job_data.get("Experience", "N/A"),
                job_data.get("Seniority", "N/A"),
                job_data.get("url", "N/A")
            ])
        logging.info(f"Appended job data for {job_data.get('url', 'unknown')} to CSV")
    except IOError as e:
        logging.error(f"IO error writing to CSV: {e}", exc_info=True)
        raise
    except Exception as e:
        logging.error(f"Unexpected error writing to CSV: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        logging.info("Starting Flask server")
        app.run(debug=True, host="0.0.0.0", port=5002)
    except KeyboardInterrupt:
        logging.info("Server stopped by user (Ctrl+C)")
    except Exception as e:
        logging.error(f"Failed to start Flask server: {e}", exc_info=True)
        raise
    finally:
        logging.info("Flask server shutdown complete")