# -*- coding: utf-8 -*-
import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import logging
import os
import json
from urllib.parse import urlparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('scraping.log')
    ]
)

# Configure Selenium
CHROME_DRIVER_PATH = "/Users/danya1/Desktop/chromedriver"
BASE_URL = "https://www.rozee.pk/job/jsearch/q/"
OUTPUT_FILE = "/Users/danya1/Desktop/solr-semantic-search/data/extracted/scrapedd.txt"

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
logging.info(f"Output file path: {os.path.abspath(OUTPUT_FILE)}")

# Set up Chrome options
options = Options()
# options.add_argument("--headless")  # Uncomment for headless mode
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-notifications")
options.add_argument("--window-size=1920,1080")
options.add_argument(
    "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
)

# Initialize the WebDriver
try:
    service = Service(CHROME_DRIVER_PATH)
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(30)
except Exception as e:
    logging.error(f"Failed to initialize WebDriver: {e}")
    raise

def scroll_page():
    """Scroll down the page to load more job listings."""
    try:
        last_height = driver.execute_script("return document.body.scrollHeight")
        for _ in range(3):  # Scroll 3 times
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)  # Wait for page to load
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
    except Exception as e:
        logging.error(f"Error scrolling page: {e}")

def normalize_url(url):
    """Normalize URL to ensure consistent format."""
    if not url:
        return ""
    
    url = url.strip()
    
    if url.startswith("//"):
        url = "https:" + url
    elif url.startswith("/"):
        url = "https://www.rozee.pk" + url
    
    # Remove any duplicate slashes in the URL (except for the protocol)
    parsed = urlparse(url)
    path = parsed.path.replace("//", "/")
    url = f"{parsed.scheme}://{parsed.netloc}{path}"
    
    if parsed.query:
        url += f"?{parsed.query}"
    
    return url

def get_job_links(search_query):
    """Fetch job listing links from Rozee.pk search results."""
    search_url = BASE_URL + search_query.replace(" ", "%20")
    logging.info(f"Searching URL: {search_url}")
    try:
        driver.get(search_url)
        # Wait for job listings to load
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.job"))
        )
        scroll_page()

        # Parse job listings
        soup = BeautifulSoup(driver.page_source, "html.parser")
        job_links = set()
        
        # Try different selectors to find job links
        job_cards = soup.select("div.job") or soup.select("div.job-listing")
        
        for job_card in job_cards:
            # Try to find link in the h3 tag first
            h3_elem = job_card.find("h3")
            if h3_elem and h3_elem.find("a", href=True):
                job_url = h3_elem.find("a", href=True)["href"]
            else:
                # Try to find any link in the job card
                link_elem = job_card.find("a", href=True)
                if not link_elem:
                    continue
                job_url = link_elem["href"]
            
            # Normalize the URL
            full_url = normalize_url(job_url)
            
            # Make sure it's a job listing URL
            if "job" in full_url.lower() and full_url.startswith("http"):
                job_links.add(full_url)

        logging.info(f"Found {len(job_links)} valid job links")
        return list(job_links)
    except Exception as e:
        logging.error(f"Error fetching job links: {e}")
        return []

def extract_job_details(soup, job_url):
    """Extract structured job details from the soup."""
    job_data = {
        "job_title": "N/A",
        "company_name": "N/A",
        "location": "N/A",
        "job_description": "N/A",
    }
    
    try:
        # Try to extract job title
        job_title_elem = (
            soup.select_one("h1.job-title") or 
            soup.select_one("div.job-title h1") or
            soup.select_one("h1.job-dtl-title") or
            soup.select_one("h1")
        )
        if job_title_elem:
            job_data["job_title"] = job_title_elem.get_text(strip=True)
        
        # Try to extract company name
        company_elem = (
            soup.select_one("div.company-name") or
            soup.select_one("span.company-name") or
            soup.select_one("a.company-name") or
            soup.select_one("div.job-company-name")
        )
        if company_elem:
            job_data["company_name"] = company_elem.get_text(strip=True)
        
        # Try to extract location
        location_elem = (
            soup.select_one("div.job-location") or
            soup.select_one("span.job-location") or
            soup.select_one("div.job-loc") or
            soup.select_one("div.location")
        )
        if location_elem:
            job_data["location"] = location_elem.get_text(strip=True)
        
        # Log the extracted structured data
        logging.info(f"Extracted structured data: {job_data}")
        
        return job_data
    except Exception as e:
        logging.error(f"Error extracting structured data: {e}")
        return job_data

def get_job_details(job_url):
    """Extract job details from a Rozee.pk job page."""
    if not job_url.startswith("http"):
        logging.error(f"Skipping invalid URL: {job_url}")
        return None

    logging.info(f"Scraping job details from: {job_url}")
    try:
        driver.get(job_url)
        
        # Wait for the job details to load
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.job-detail-container, div.job-detail, div.job-dtl"))
        )
        
        # Give the page a moment to fully render
        time.sleep(3)
        
        # Get the page source
        soup = BeautifulSoup(driver.page_source, "html.parser")
        
        # Extract structured job data
        structured_data = extract_job_details(soup, job_url)
        
        # Find the job description container
        job_container = (
            soup.select_one("div.job-detail-container") or
            soup.select_one("div.boxb.job-dtl") or
            soup.select_one("div.job-detail") or
            soup.select_one("div#job-detail") or
            soup.select_one("div.job-description") or
            soup.select_one("div.job-details") or
            soup.select_one("div.content")
        )
        
        if not job_container:
            logging.warning(f"No job content container found for {job_url}")
            # Save the page source for debugging
            with open(f"failed_{job_url.split('/')[-1]}.html", "w", encoding="utf-8") as f:
                f.write(driver.page_source)
            logging.info(f"Saved failed page source to failed_{job_url.split('/')[-1]}.html")
            return None

        # Remove script and style tags
        for unwanted in job_container.select("script, style, iframe, noscript"):
            unwanted.decompose()
        
        # Extract text with proper spacing
        job_text = job_container.get_text(separator="\n", strip=True)
        
        # Clean up the text
        job_text = "\n".join(line.strip() for line in job_text.split("\n") if line.strip())

        if not job_text:
            logging.warning(f"No text extracted from {job_url}")
            return None

        logging.info(f"Extracted text for {job_url}: {job_text[:100]}...")

        # Save the extracted text to file
        try:
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                f.write(f"URL: {job_url}\n")
                f.write(f"Job Title: {structured_data['job_title']}\n")
                f.write(f"Company: {structured_data['company_name']}\n")
                f.write(f"Location: {structured_data['location']}\n")
                f.write(f"Job Description:\n{job_text}\n")
                f.write("-" * 80 + "\n")
                f.flush()  # Force write to disk
            logging.info(f"Saved scraped text for {job_url} to {OUTPUT_FILE}")
        except IOError as e:
            logging.error(f"Failed to write to {OUTPUT_FILE}: {e}")
            return None

        # Prepare the job data to be sent to the API
        # Include the structured data we extracted
        formatted_job_text = f"""
Job Title: {structured_data['job_title']}
Company Name: {structured_data['company_name']}
Location: {structured_data['location']}

Job Description:
{job_text}
"""

        return {
            "url": job_url, 
            "full_text": formatted_job_text
        }
    except Exception as e:
        logging.error(f"Error scraping job details for {job_url}: {e}")
        return None

def post_job_to_api(job_data):
    """Send job data to the processing API."""
    api_url = "http://localhost:5002/process_job"
    try:
        # Format the data for the API
        api_data = {
            "job_text": job_data["full_text"],
            "job_url": job_data["url"]
        }
        
        # Log the data being sent
        logging.info(f"Sending data to API: URL={api_data['job_url']}, Text length={len(api_data['job_text'])}")
        logging.debug(f"Text preview: {api_data['job_text'][:100]}...")
        
        # Send the request
        response = requests.post(api_url, json=api_data, timeout=30)
        
        # Log the response
        if response.status_code == 200:
            logging.info(f"Successfully posted job: {api_data['job_url']}")
            logging.debug(f"API response: {response.json()}")
        else:
            logging.error(f"API error: {response.status_code} - {response.text}")
            
    except requests.exceptions.ConnectionError:
        logging.error(f"Failed to connect to API at {api_url}. Is the processor running?")
    except requests.exceptions.Timeout:
        logging.error(f"Request to {api_url} timed out for job: {job_data['url']}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to post job: {job_data['url']} - Error: {e}")
        if 'response' in locals():
            logging.error(f"Status code: {response.status_code}")
    except Exception as e:
        logging.error(f"Unexpected error posting job: {job_data['url']} - {e}")

if __name__ == "__main__":
    try:
        # Clear the file at the start to ensure fresh data
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write("")
        logging.info(f"Cleared {OUTPUT_FILE} at start")

        # List of job queries to search for
        job_queries = [
            "Software Engineer",
            "Data Scientist",
            "Project Manager",
            "Mechanical Engineer",
            "Electrical Engineer"
        ]
        
        all_job_links = []
        
        # Get job links for each query
        for query in job_queries:
            logging.info(f"Searching for '{query}' jobs")
            job_links = get_job_links(query)
            all_job_links.extend(job_links)
            
            # Small delay between searches
            time.sleep(2)
        
        # Remove duplicates
        all_job_links = list(set(all_job_links))
        logging.info(f"Found {len(all_job_links)} unique job links across all queries")
        
        # Process each job link
        for i, job_url in enumerate(all_job_links[:20]):  # Limit to 20 jobs for testing
            logging.info(f"Processing job {i+1}/{min(20, len(all_job_links))}: {job_url}")
            
            # Get job details
            details = get_job_details(job_url)
            
            if details:
                # Post the job data to the API
                post_job_to_api(details)
            
            # Delay between job scrapes to avoid overloading the server
            time.sleep(2)

        # Save the final page source for debugging
        with open("debug.html", "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        logging.info("Saved raw HTML for debugging.")
        
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        # Save the error page for debugging
        try:
            with open("error.html", "w", encoding="utf-8") as f:
                f.write(driver.page_source)
            logging.info("Saved error page HTML for debugging.")
        except:
            pass
    finally:
        driver.quit()
        logging.info("Script execution completed.")