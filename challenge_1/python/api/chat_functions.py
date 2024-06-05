import dotenv
from openai import AzureOpenAI
import os

def call_bike_repair():
    dotenv.load_dotenv()
    
    search_endpoint = os.getenv("SEARCH_ENDPOINT")
    search_index = os.getenv("SEARCH_INDEX")
    search_key = os.getenv("SEARCH_KEY")

    client = AzureOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2024-02-15-preview",
    azure_deployment="turbogpt"
    )
    
    client.chat.completions.create(
        model="turbogpt", # model = "deployment_name".
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message}
        ],
        extra_body={
        "data_sources": [
            {
                "type": "azure_search",
                "parameters": {
                    "endpoint": search_endpoint,
                    "index_name": search_index,
                    "authentication": {
                        "type": "api_key",
                        "key": search_key
                    }
                }
            }
        ]
        },
        temperature= 0,
        top_p= 1,
        max_tokens= 800,
        stop=None,
        stream=False
    )