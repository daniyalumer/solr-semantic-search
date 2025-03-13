import os
import pandas as pd
import json
import datetime
from processing.extract import process_files, get_base_filenames
from processing.parse_cv import process_cvs
from processing.calculate_embeddings import embed_json_files
from clients.solr import create_solr_client_cv, create_solr_client_profile
from indexing import index_documents, delete_index


def main():

    # Process CV dataset (PDF, DOC, DOCX)
    cv_dir = "data/dataset/CV"
    cv_output_csv = "data/extracted/cv.csv"
    if not os.path.exists(cv_output_csv):
        process_files(cv_dir, cv_output_csv, file_types=['pdf', 'doc', 'docx'])
    else:
        print(f"{cv_output_csv} already exists. Skipping processing.")

    # Get base filenames from CV directory
    cv_base_filenames = get_base_filenames(cv_dir, '-CV')
    print(f"Found {len(cv_base_filenames)} base filenames in CV directory.")

    # Process PROFILE dataset (PDF, DOC, DOCX)
    profile_dir = "data/dataset/PROFILE"
    profile_output_csv = "data/extracted/profile.csv"
    if not os.path.exists(profile_output_csv):
        process_files(profile_dir, profile_output_csv, file_types=['pdf', 'doc', 'docx'], filter_set=cv_base_filenames)
    else:
        print(f"{profile_output_csv} already exists. Skipping processing.")

    # csv_path = "data/extracted/cv.csv"
    # batch_size = 1000  # You can adjust the batch size as needed
    # process_cvs(csv_path, "data/parsed_data/cv", batch_size)

    # csv_path = "data/extracted/profile.csv"
    # batch_size = 1000  # You can adjust the batch size as needed
    # process_cvs(csv_path, "data/parsed_data/profile", batch_size)

    # # Calculate embeddings for CV documents
    # input_directory = 'data/parsed_data/cv'
    # output_directory = 'data/parsed_data_embeddings/cv'
    # embed_json_files(input_directory, output_directory)

    # # Calculate embeddings for Profile documents
    # input_directory = 'data/parsed_data/profile'
    # output_directory = 'data/parsed_data_embeddings/profile'
    # embed_json_files(input_directory, output_directory)

    client_cv = create_solr_client_cv()
    print(client_cv.ping())

    client_profile = create_solr_client_profile()
    print(client_profile.ping())
    
    print("--------------------------DELETING COLLECTION-------------------------")
    delete_index(client_cv)
    delete_index(client_profile)

    print("--------------------------INDEXING DOCUMENTS-------------------------")

    # Index CV documents
    data_directory = "data/parsed_data_embeddings/cv"
    index_documents(client_cv, data_directory)

    # Index Profile documents
    data_directory = "data/parsed_data_embeddings/profile"
    index_documents(client_profile, data_directory)

    print("---------------------------Indexing complete--------------------------")


if __name__ == "__main__":
    main()
