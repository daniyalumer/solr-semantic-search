## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/rozee-gpt-semantic-solr.git
    cd rozee-gpt-semantic-solr
    ```

2. Install the required dependencies using `pipenv`:

    ```bash
    pipenv install
    ```

3. Activate the virtual environment:

    ```bash
    pipenv shell
    ```

## Usage

1. Configure the Solr endpoint and API key in the `config.py` file.

2. Run the main script to process and ingest data:

    ```bash
    pipenv run python main.py
    ```

3. Alternatively, you can run the `ingest.py` script to ingest data only:

    ```bash
    pipenv run python ingest.py
    ```

4. To query the ingested data, use the `query.py` script:

    ```bash
    pipenv run python query.py
    ```

## Configuration

Update the `config.py` file with your Solr endpoint and API key:

```python
SOLR_ENDPOINT = "https://your-solr-endpoint"
SOLR_API_KEY = "your-api-key"
```