import json
import os

def index_documents(client, data_directory):
    for filename in os.listdir(data_directory):
        if filename.endswith('.json'):
            file_path = os.path.join(data_directory, filename)
            print(f"Indexing file: {filename}")  # Print the name of the file being indexed
            with open(file_path, 'r') as f:
                content = f.read()
                doc = json.loads(content)  # Parse the JSON content
                client.add([doc])  # Add the document to Solr

def delete_index(client):
    # Solr does not support deleting an index directly, you can delete all documents instead
    client.delete(q='*:*')
    print(f"Deleted all documents in collection: {client}")