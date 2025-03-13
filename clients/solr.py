import logging
import pysolr
from config.config import SOLR_ENDPOINT, SOLR_PROFILE

def create_solr_client_cv():
    logging.info("Creating Solr client...")
    client = pysolr.Solr(SOLR_ENDPOINT, timeout=120)
    logging.info("Solr client created.")
    return client

def create_solr_client_profile():
    logging.info("Creating Solr client...")
    client = pysolr.Solr(SOLR_PROFILE, timeout=120)
    logging.info("Solr client created.")
    return client