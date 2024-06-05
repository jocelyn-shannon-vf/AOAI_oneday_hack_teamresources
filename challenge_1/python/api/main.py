from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles
import requests
import dotenv
import os
from openai import AzureOpenAI
from array import array
from PIL import Image, ImageDraw
import sys
import time
from matplotlib import pyplot as plt
import numpy as np

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

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

def analyzeImage():

    image_file = 'images/person.jpg'
    
    
    cog_endpoint = os.getenv('COG_SERVICE_ENDPOINT')
    cog_key = os.getenv('COG_SERVICE_KEY')

    credential = CognitiveServicesCredentials(cog_key) 
    cv_client = ComputerVisionClient(cog_endpoint, credential)


    print('Analyzing', image_file)

    # Specify features to be retrieved
    features = [VisualFeatureTypes.description,
                VisualFeatureTypes.tags,
                VisualFeatureTypes.categories,
                VisualFeatureTypes.brands,
                VisualFeatureTypes.objects,
                VisualFeatureTypes.adult]
    

    # Get image analysis
    with open(image_file, mode="rb") as image_data:
        print("reaching point check")
        analysis = cv_client.analyze_image_in_stream(image_data , features)
        print(analysis)

    # Get image description
    for caption in analysis.description.captions:
        print("Description: '{}' (confidence: {:.2f}%)".format(caption.text, caption.confidence * 100))


    tagList = list()
    
    # Get image tags
    if (len(analysis.tags) > 0):
        print("Tags: ")
        for tag in analysis.tags:
            tagList.append(tag.name)
            print(" -'{}' (confidence: {:.2f}%)".format(tag.name, tag.confidence * 100))
            
    # Get image categories
    if (len(analysis.categories) > 0):
        print("Categories:")
        landmarks = []
        for category in analysis.categories:
            # Print the category
            print(" -'{}' (confidence: {:.2f}%)".format(category.name, category.score * 100))
            if category.detail:
                # Get landmarks in this category
                if category.detail.landmarks:
                    for landmark in category.detail.landmarks:
                        if landmark not in landmarks:
                            landmarks.append(landmark)

        # If there were landmarks, list them
        if len(landmarks) > 0:
            print("Landmarks:")
            for landmark in landmarks:
                print(" -'{}' (confidence: {:.2f}%)".format(landmark.name, landmark.confidence * 100))


    # Get brands in the image
    if (len(analysis.brands) > 0):
        print("Brands: ")
        for brand in analysis.brands:
            print(" -'{}' (confidence: {:.2f}%)".format(brand.name, brand.confidence * 100))
    
    # Get objects in the image
    if len(analysis.objects) > 0:
        print("Objects in image:")

        # Prepare image for drawing
        fig = plt.figure(figsize=(8, 8))
        plt.axis('off')
        image = Image.open(image_file)
        draw = ImageDraw.Draw(image)
        color = 'cyan'
        for detected_object in analysis.objects:
            # Print object name
            print(" -{} (confidence: {:.2f}%)".format(detected_object.object_property, detected_object.confidence * 100))
            
            # Draw object bounding box
            r = detected_object.rectangle
            bounding_box = ((r.x, r.y), (r.x + r.w, r.y + r.h))
            draw.rectangle(bounding_box, outline=color, width=3)
            plt.annotate(detected_object.object_property,(r.x, r.y), backgroundcolor=color)
        # Save annotated image
        plt.imshow(image)
        outputfile = 'objects.jpg'
        fig.savefig(outputfile)
        print('  Results saved in', outputfile)

    # Get moderation ratings
    ratings = 'Ratings:\n -Adult: {}\n -Racy: {}\n -Gore: {}'.format(analysis.adult.is_adult_content,
                                                                        analysis.adult.is_racy_content,
                                                                        analysis.adult.is_gory_content)
    print(tagList)
    return tagList

def chat_method(message:str):
    dotenv.load_dotenv()

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
        temperature= 0,
        top_p= 1,
        max_tokens= 800,
        stop=None,
        stream=False
    )
    print(response.choices[0].message.content)

    return response


def call_bike_repair(message: str):
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

    return response.choices[0].message.content

def sql_query(question):
    from db_connected_bot import MSSQL_AGENT_PREFIX, sql_llm_connection

    response = sql_llm_connection(question)
    print(type(response))
    print(response)
    print(response['output'])
    return response['output']


import inspect


# helper method used to check if the correct arguments are provided to a function
def check_args(function, args):
    sig = inspect.signature(function)
    params = sig.parameters

    # Check if there are extra arguments
    for name in args:
        if name not in params:
            return False
    # Check if the required arguments are provided
    for name, param in params.items():
        if param.default is param.empty and name not in args:
            return False

    return True

import json

def run_conversation(messages, tools, available_functions):
    # Step 1: send the conversation and available functions to GPT
    client = AzureOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2024-02-15-preview",
    azure_deployment="turbogpt"
    )
    
    response = client.chat.completions.create(
        model="turbogpt",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    response_message = response.choices[0].message
    print(response_message)
    # Step 2: check if GPT wanted to call a function
    if response_message.tool_calls:
        print("Recommended Function call:")
        print(response_message.tool_calls[0])
        print()

        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors

        function_name = response_message.tool_calls[0].function.name

        # verify function exists
        if function_name not in available_functions:
            return "Function " + function_name + " does not exist"
        function_to_call = available_functions[function_name]

        # verify function has correct number of arguments
        function_args = json.loads(response_message.tool_calls[0].function.arguments)
        if check_args(function_to_call, function_args) is False:
            return "Invalid number of arguments for function: " + function_name
        function_response = function_to_call(**function_args)

        print("Output of function call:")
        print(function_response)
        print()

        # Step 4: send the info on the function call and function response to GPT

        # adding assistant response to messages
        messages.append(
            {
                "role": response_message.role,
                "function_call": {
                    "name": response_message.tool_calls[0].function.name,
                    "arguments": response_message.tool_calls[0].function.arguments,
                },
                "content": None,
            }
        )

        # adding function response to messages
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )  # extend conversation with function response

        print("Messages in second request:")
        for message in messages:
            print(message)
        print()

        second_response = client.chat.completions.create(
            messages=messages,
            model="turbogpt",
        )  # get a new response from GPT where it can see the function response

        return second_response
    else:
        return response_message.content
    


# chatbot API to be extended with OpenAI code
@app.post("/chat")
async def chat(request: Request):

    json = await request.json()
    print(json)

    dotenv.load_dotenv()
    tools = [
    {
        "type": "function",
        "function": {
            "name": "chat_method",
            "description": "Getting general knowledge",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Message to be sent to the LLM",
                    }
                },
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_bike_repair",
            "description": "Getting information to assist a bike repair",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Message to be sent to the LLM",
                    },
                    }
                },
                "required": ["message"],
            },
        },
    ]

    available_functions = {
        "chat_method": chat_method,
        "call_bike_repair": call_bike_repair
    }

    messages=[
            {"role": "system", "content": "You are a helpful assistant. You have access to Azure AI Search to find information on bike repairs."},
            {"role": "user", "content": json['message']}
        ]
    tool_choice = "auto"

    response = run_conversation(messages, tools=tools, available_functions=available_functions)

    print(response)
    return response
    #response = chat_method(json['message'])
    #response = sql_query(json['message'])

    #response.choices[0].message.content
    
    #return {'message': response }
    #return {"message": response.choices[0].message.content}

# Image generattion API to be extended with OpenAI code
@app.post("/generateImage")
async def generateImage(request: Request):
    json = await request.json()
    face = str(json)
    #{'face': 'round', 'eyebrows': 'thick', 'eye_color': 'blue', 'facial_features': 'strong jaw', 'hair': 'long green', 'clothes': 'pijamas', 'height': '3 foot', 'weight': 'heavy', 'other_apparel': ''} 
    print(json)

    dotenv.load_dotenv()
    tagList = analyzeImage()

    dalle_request = f"""Generate me a prompt that I can send to the DALL-E model that will create 
    a photorealistic avatar image with the following attributes: {tagList}"""

    response = chat_method(dalle_request)

    #new_request = request(json={"message",f"{dalle_request}"})
    #response=await chat(dalle_request)

    # Get Azure OpenAI Service settings
    dotenv.load_dotenv()
    api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
    api_version="2024-02-15-preview"
   
    # Get prompt for image to be generated
    prompt = response.choices[0].message.content
    # Call the DALL-E model
    url = "{}openai/deployments/dalle3/images/generations?api-version={}".format(api_base, api_version)
    headers= { "api-key": api_key, "Content-Type": "application/json" }
    body = {
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024"
    }
    response = requests.post(url, headers=headers, json=body)
    print(response)
    # Get the revised prompt and image URL from the response
    revised_prompt = response.json()['data'][0]['revised_prompt']
    image_url = response.json()['data'][0]['url']
    # Display the URL for the generated image
    print(revised_prompt)
    print(image_url)
    return {"url": image_url}

app.mount("/", StaticFiles(directory="public"), name="ui")