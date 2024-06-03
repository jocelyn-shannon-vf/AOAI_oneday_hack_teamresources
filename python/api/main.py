from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles

import dotenv
import os
from openai import AzureOpenAI

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# serve static files
@app.get("/")
async def main():
    return FileResponse("public/index.html")

# chatbot API to be extended with OpenAI code
@app.post("/chat")
async def chat(request: Request):
    json = await request.json()
    print(json)

    dotenv.load_dotenv()

    import os
    from openai import AzureOpenAI

    search_endpoint = os.getenv("SEARCH_ENDPOINT")
    search_index = os.getenv("SEARCH_INDEX")
    search_key = os.getenv("SEARCH_KEY")
    
    client = AzureOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2024-02-15-preview",
    azure_deployment="turbogpt"
    )

    response = client.chat.completions.create(
        model="turbogpt", # model = "deployment_name".
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": json["message"]}
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
        }
    )

    response.choices[0].message.content

    return {"message": response.choices[0].message.content}

# Image generattion API to be extended with OpenAI code
@app.post("/generateImage")
async def generateImage(request: Request):
    json = await request.json()
    print(json)

    ############################
    ### Add OpenAI code here ###
    ############################

    return {"url": "https://via.placeholder.com/100"}

app.mount("/", StaticFiles(directory="public"), name="ui")