import requests
import json
import pandas as pd
import os
from datetime import datetime
import numpy as np

def vector_to_str(vector):
    """Convert a vector array to a comma-separated string"""
    return ",".join([str(val) for val in vector])

def load_job_embeddings(csv_file_path, row_index):
    """
    Load job embeddings from a CSV file for a specific row
    
    Parameters:
    csv_file_path (str): Path to the CSV file with embeddings
    row_index (int): Index of the row to load (matching the actual row number in the CSV)
    
    Returns:
    tuple: (job_title_vector, skills_vector, location_vector, desc_vector, job_info)
    """
    try:
        # Read the CSV file containing embeddings
        df = pd.read_csv(csv_file_path)
        
        # Use row_index as the actual row number (2-based, where 2 is the first data row)
        # Adjust pandas_index to be 0-based (subtract 2 instead of 1)
        pandas_index = row_index - 2
        
        # Validate the converted index
        if pandas_index < 0 or pandas_index >= len(df):
            raise ValueError(f"Row {row_index} is out of bounds. CSV has rows 2-{len(df)+1} (where 2 is the first data row).")
        
        # Extract all vectors
        job_title_vector = eval(df.iloc[pandas_index]['title_vector'])
        
        skills_vector = None
        location_vector = None
        desc_vector = None
        
        # Collect job info for display
        job_info = {
            "job_title": df.iloc[pandas_index]['Job Title'],
            "seniority": df.iloc[pandas_index]['Seniority'].lower() if not pd.isna(df.iloc[pandas_index]['Seniority']) else None
        }
        
        # Extract optional vectors if available
        if 'skills_vector' in df.columns:
            skills_vector = eval(df.iloc[pandas_index]['skills_vector'])
            job_info["skills"] = df.iloc[pandas_index]['Required Skills']
            
        if 'location_vector' in df.columns:
            location_vector = eval(df.iloc[pandas_index]['location_vector'])
            job_info["location"] = df.iloc[pandas_index]['Location']
            
        if 'desc_vector' in df.columns:
            desc_vector = eval(df.iloc[pandas_index]['desc_vector'])
            desc = df.iloc[pandas_index]['Job Description']
            job_info["description"] = desc
        
        # Print information about the selected job
        print(f"Selected Job: {job_info['job_title']}")
        if 'skills' in job_info:
            print(f"Skills: {job_info['skills']}")
        if 'location' in job_info:
            print(f"Location: {job_info['location']}")
        if 'description' in job_info:
            # Print only first 150 chars of description to keep output manageable
            print(f"Description: {job_info['description']}")#[:150]}..." if len(job_info['description']) > 150 else f"Description: {job_info['description']}")
        if 'seniority' in job_info:
            print(f"Seniority: {job_info['seniority']}")
        
            
        return job_title_vector, skills_vector, location_vector, desc_vector, job_info
    
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return None, None, None, None, None

def build_search_query(job_title_vector, skills_vector=None, desc_vector=None, location_vector=None, job_info=None):
    """
    Build query parameters for searching with multiple vectors with different weights
    and keyword search for location and seniority as filters
    """
    # Convert job title vector to string (this one is mandatory)
    job_title_vector_str = vector_to_str(job_title_vector)
    
    # Create query parameters with job title vector
    query_params = {
        "defType": "edismax",
        "q": "*:*",
        "bq": [
           f"{{!knn f=work_experience_job_titles_embedding topK=50 boost=4}}[{job_title_vector_str}]"
        ],
        "fl": "document_id, work_experience_job_titles, skills, contact_information_address, work_experience_descriptions, work_experience_seniority, score",
        "rows": 20,
        "fq": ["score:[1.01 TO *]"],
    }
    
    # Initialize filter queries list
    filter_queries = []
    
    # Add skills vector to query if available (weight 3)
    if skills_vector:
        skills_vector_str = vector_to_str(skills_vector)
        query_params["bq"].append(
            f"{{!knn f=skills_embedding topK=50 boost=3}}[{skills_vector_str}]"
        )
    
    # Add location as a filter query if available in job_info
    if job_info and "location" in job_info:
        location = job_info["location"].strip()
        if location:
            # Split location by commas and clean up each part
            location_parts = [part.strip() for part in location.split(',')]
            
            # The last part is typically the country
            country = location_parts[-1] if location_parts else ""
            
            # City could be in multiple parts for complex locations
            cities = []
            # Extract cities - all parts except the country
            if len(location_parts) > 1:
                cities = location_parts[:-1]
            elif len(location_parts) == 1:
                # If only one part, assume it's a city
                cities = [location_parts[0]]
                
            print(f"Extracted - Cities: {cities}, Country: {country}")
            
            # Build location filter queries
            location_filters = []
            
            # # Add country filter if available
            # if country and len(country.strip()) > 1:
            #     location_filters.append(f"contact_information_address:*{country}*")
            
            # Add city filters - join with OR for multiple cities
            if cities:
                city_filters = []
                for city in cities:
                    if len(city.strip()) > 1:  # Avoid very short terms
                        city_filters.append(f"contact_information_address:*{city}*")
                
                if city_filters:
                    location_filters.append(f"({' OR '.join(city_filters)})")
            
            # Add location filters to main filter queries
            if location_filters:
                filter_queries.extend(location_filters)

    # Lets add a semantic query for location for testing purposes
    # if location_vector:
    #     location_vector_str = vector_to_str(location_vector)
    #     query_params["bq"].append(
    #         f"{{!knn f=contact_information_address_embedding topK=5 boost=15}}[{location_vector_str}]"
    #     )
    
    # Add seniority as a filter query if available in job_info
    if job_info and "seniority" in job_info and job_info["seniority"]:
        seniority = job_info["seniority"].strip().lower()
        filter_queries.append(f"work_experience_seniority:\"{seniority}\"")

    # Add description vector to query if available (weight 2)
    if desc_vector:
        desc_vector_str = vector_to_str(desc_vector)
        query_params["bq"].append(
            f"{{!knn f=work_experience_descriptions_embedding topK=30 boost=2}}[{desc_vector_str}]"
        )
    
    # Add all filter queries if any exist
    if filter_queries:
        query_params["fq"] = filter_queries
    
    return query_params

def search_collection_by_vectors(collection_name, query_params):
    """
    Search for documents in the specified collection using the given query parameters
    """
    # Save query parameters to file for debugging
    save_dir = "data/generated_query"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    query_file = os.path.join(save_dir, f"query_{collection_name}_{timestamp}.json")
    
    with open(query_file, 'w') as f:
        json.dump(query_params, f, indent=2)
    print(f"Query for {collection_name} saved to {query_file}")
    
    # Send the request
    response = requests.post(
        f"https://ss251402-c5dglu8c-ap-south-1-aws.searchstax.com/solr/{collection_name}/select",
        data=query_params
    )
    
    return response

def display_results(result, collection_name):
    """Display search results in a formatted manner"""
    print(f"\n{collection_name.upper()} Search Results:")
    print("=" * 80)
    
    if 'response' in result and 'docs' in result['response']:
        if len(result['response']['docs']) == 0:
            print(f"No results found in {collection_name}")
        
        for i, doc in enumerate(result['response']['docs']):
            # Display the result
            print(f"{i+1}. Document ID: {doc['document_id'][0]}")
            print(f"   Score: {doc.get('score', 'N/A')}")
            
            if 'work_experience_job_titles' in doc:
                job_titles = ', '.join(doc['work_experience_job_titles'])
                print(f"   Job Titles: {job_titles}")

            if 'work_experience_seniority' in doc:
                seniority = doc['work_experience_seniority']
                print(f"   Seniority: {seniority}")
            
            if 'skills' in doc:
                skills = ', '.join(doc['skills'])
                print(f"   Skills: {skills}")
            
            if 'contact_information_address' in doc:
                location = doc['contact_information_address']
                print(f"   Location: {location}")
            
            if 'work_experience_descriptions' in doc:
                summary = doc['work_experience_descriptions']
                print(f"  Work Experience Descriptions: {summary}")

            print("-" * 80)
    else:
        print("No search results found or unexpected response format.")
        print(result)

# Main execution
if __name__ == "__main__":
    # Get row index from user input (2-based, where 2 is the first data row)
    try:
        row_index = int(input("Enter the row number from the CSV (2 is the first data row): "))
        if row_index < 2:
            print("Row numbers start at 2. Using row 2.")
            row_index = 2
    except ValueError:
        print("Invalid input. Using default row 2.")
        row_index = 2
    
    # Load vectors from CSV
    csv_path = "data/rozee_jd/rozee_jobs_with_embeddings.csv"
    job_title_vector, skills_vector, location_vector, desc_vector, job_info = load_job_embeddings(csv_path, row_index)
    
    if job_title_vector:
        # Build query parameters
        query_params = build_search_query(
            job_title_vector, 
            skills_vector, 
            desc_vector,
            location_vector,
            job_info
        )
        
        # Search CV collection
        print("\nSearching cv_collection...")
        cv_response = search_collection_by_vectors("cv_collection", query_params)
        
        # Search profile collection
        print("\nSearching profile_collection collection...")
        profile_response = search_collection_by_vectors("profile_collection", query_params)
        
        # Process and display results for CV collection
        try:
            cv_result = cv_response.json()
            display_results(cv_result, "cv_collection")
        except Exception as e:
            print(f"Error processing cv_collection search results: {e}")
            print(cv_response.text)
            
        # Process and display results for Profile collection
        try:
            profile_result = profile_response.json()
            display_results(profile_result, "profile_collection")
        except Exception as e:
            print(f"Error processing cv_profile search results: {e}")
            print(profile_response.text)
    else:
        print("Failed to load embeddings. Exiting.")