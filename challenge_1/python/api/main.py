from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles
import requests
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

def chat_method(message:str):
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

    response = client.chat.completions.create(
        model="turbogpt", # model = "deployment_name".
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message}
        ],
        #extra_body={
        #"data_sources": [
        #    {
        #        "type": "azure_search",
        #        "parameters": {
        #            "endpoint": search_endpoint,
        #            "index_name": search_index,
        #            "authentication": {
        #                "type": "api_key",
        #                "key": search_key
        #            }
        #        }
        #    }
        #]
        #},
        temperature= 0,
        top_p= 1,
        max_tokens= 800,
        stop=None,
        stream=False
    )
    print(response.choices[0].message.content)

    return response


# chatbot API to be extended with OpenAI code
@app.post("/chat")
async def chat(request: Request):
    json = await request.json()
    print(json)
    #dotenv.load_dotenv()

    response = chat_method(json['message'])

    response.choices[0].message.content

    return {"message": response.choices[0].message.content}

# Image generattion API to be extended with OpenAI code
@app.post("/generateImage")
async def generateImage(request: Request):
    json = await request.json()
    face = str(json)
    #{'face': 'round', 'eyebrows': 'thick', 'eye_color': 'blue', 'facial_features': 'strong jaw', 'hair': 'long green', 'clothes': 'pijamas', 'height': '3 foot', 'weight': 'heavy', 'other_apparel': ''} 
    print(json)

    dalle_request = f"""Generate me a prompt that I can send to the DALL-E model that will create 
    a photorealistic avatar image with the following attributes: {face}"""

    response = chat_method(dalle_request)

    #new_request = request(json={"message",f"{dalle_request}"})

    #response=await chat(dalle_request)
    ## Get Azure OpenAI Service settings
    #dotenv.load_dotenv()
    #api_base = os.getenv("AZURE_OPENAI_ENDPOINT"), 
    #api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    #api_version="2024-02-15-preview",
   # 
    ## Get prompt for image to be generated
    #prompt = json

    ## Call the DALL-E model
    #url = "{}openai/deployments/dalle3/images/generations?api-version={}".format(api_base, api_version)
    #headers= { "api-key": api_key, "Content-Type": "application/json" }
    #body = {
    #    "prompt": prompt,
    #    "n": 1,
    #    "size": "1024x1024"
    #}
    #response = requests.post(url, headers=headers, json=body)
    #print(response)
    ## Get the revised prompt and image URL from the response
    #revised_prompt = response.json()['data'][0]['revised_prompt']
    #image_url = response.json()['data'][0]['url']

    ## Display the URL for the generated image
    #print(revised_prompt)
    #print(image_url)

    #return {"url": image_url}
    #image_url = response.json()['data'][0]['url']
    print(response)
    return response

app.mount("/", StaticFiles(directory="public"), name="ui")