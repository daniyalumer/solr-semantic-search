# -*- coding: utf-8 -*-
import time
import logging
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from processing import process_job_function  # Import the function from processing.py
from rozee_embeddings import calculate_embeddings  # Import the embedding function

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
BASE_URL = "https://www.rozee.pk/top-jobs"
OUTPUT_FILE = "/Users/danya1/Desktop/solr-semantic-search/data/extracted_jd/scrapedd.txt"
CSV_FILE = "/Users/danya1/Desktop/solr-semantic-search/data/extracted_jd/rozee_jobs_llm.csv"
EMBEDDINGS_FILE = "/Users/danya1/Desktop/solr-semantic-search/data/extracted_jd/rozee_jobs_with_embeddings.csv"

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
logging.info(f"Output file path: {os.path.abspath(OUTPUT_FILE)}")

# Set up Chrome options
options = Options()
# Uncomment the following line to run in headless mode
# options.add_argument("--headless")
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
    parsed = urlparse(url)
    path = parsed.path.replace("//", "/")
    url = f"{parsed.scheme}://{parsed.netloc}{path}"
    if parsed.query:
        url += f"?{parsed.query}"
    return url

def get_job_links():
    """Fetch job listing links from the Top Jobs section on Rozee.pk."""
    logging.info(f"Searching URL: {BASE_URL}")
    try:
        driver.get(BASE_URL)
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.section.Tjbs.opages"))  # Wait for the Top Jobs section
        )
        scroll_page()
        soup = BeautifulSoup(driver.page_source, "html.parser")
        job_links = set()

        # Select the Top Jobs section
        top_jobs_section = soup.select_one("div.section.Tjbs.opages")
        if not top_jobs_section:
            logging.warning("No Top Jobs section found on the page.")
            return []

        # Extract job links from the Top Jobs section
        job_cards = top_jobs_section.select("div.col-lg-4.col-md-6")
        for job_card in job_cards:
            link_elem = job_card.find("a", href=True, class_="full_link")
            if not link_elem:
                continue
            job_url = link_elem["href"]
            full_url = normalize_url(job_url)
            if "job" in full_url.lower() and full_url.startswith("http"):
                job_links.add(full_url)

        logging.info(f"Found {len(job_links)} valid job links in the Top Jobs section")
        return list(job_links)
    except Exception as e:
        logging.error(f"Error fetching job links from Top Jobs: {e}")
        return []

def get_job_details(job_url):
    """Extract job details from a Rozee.pk job page."""
    if not job_url.startswith("http"):
        logging.error(f"Skipping invalid URL: {job_url}")
        return None
    logging.info(f"Scraping job details from: {job_url}")
    try:
        driver.get(job_url)
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.job-detail-container, div.job-detail, div.job-dtl"))
        )
        time.sleep(3)
        soup = BeautifulSoup(driver.page_source, "html.parser")
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
            return None
        for unwanted in job_container.select("script, style, iframe, noscript"):
            unwanted.decompose()
        job_text = job_container.get_text(separator="\n", strip=True)
        job_text = "\n".join(line.strip() for line in job_text.split("\n") if line.strip())
        if not job_text:
            logging.warning(f"No text extracted from {job_url}")
            return None
        logging.info(f"Extracted text for {job_url}: {job_text[:100]}...")
        try:
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                f.write(f"URL: {job_url}\n")
                f.write(f"Extracted Text:\n{job_text}\n")
                f.write("-" * 80 + "\n")
            logging.info(f"Saved extracted text for {job_url} to {OUTPUT_FILE}")
        except IOError as e:
            logging.error(f"Failed to write to {OUTPUT_FILE}: {e}")
            return None
        return {
            "url": job_url,
            "full_text": job_text
        }
    except Exception as e:
        logging.error(f"Error scraping job details for {job_url}: {e}")
        return None

if __name__ == "__main__":
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write("")
        logging.info(f"Cleared {OUTPUT_FILE} at start")
        all_job_links = get_job_links()
        logging.info(f"Found {len(all_job_links)} unique job links in the Top Jobs section")
        for i, job_url in enumerate(all_job_links[:20]):
            logging.info(f"Processing job {i+1}/{min(20, len(all_job_links))}: {job_url}")
            details = get_job_details(job_url)
            if details:
                result = process_job_function(details["full_text"], details["url"])
                if "error" in result:
                    logging.error(f"Error processing job: {result['error']}")
                else:
                    logging.info(f"Successfully processed job: {result['message']}")
            time.sleep(2)

        # Call the embedding function after processing jobs
        calculate_embeddings(CSV_FILE, EMBEDDINGS_FILE)

    except Exception as e:
        logging.error(f"Error in main execution: {e}")
    finally:
        driver.quit()
        logging.info("Script execution completed.")