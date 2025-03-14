## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/solr-semantic-search.git
    cd solr-semantic-search
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
    OR

    ```bash
    pipenv run python ingest.py
    ```

3. To query the ingested data, use the `query.py` script:

    ```bash
    pipenv run python query.py
    ```

## Configuration

Update the env file with your Solr endpoint and API key:

```g
SOLR_ENDPOINT = "https://your-solr-endpoint"
SOLR_API_KEY = "your-api-key"
```