#import agent
from agent import agent

import warnings, asyncio, os, uuid
warnings.filterwarnings('ignore')

# FastAPI related
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile


os.makedirs('temp', exist_ok=True)

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*']
    )
]

app = FastAPI(middleware=middleware, version='v0.1', title='Personal Chatbot',
                description="<b>Chatbot to answer user queries</b></br></br>")

#default API
@app.get("/")
def home():
    return {'type': 'Langchain based Chatbot'}


#API to chat with chatbot
@app.post('/chat')
async def chat(user_question, file: UploadFile=File(None)):  #takes user question and uploaded image file
    await asyncio.sleep(0.001)
    try:
        #if image is uploaded
        file_suffix = "".join(file.filename.partition(".")[1:])
        print("Name", file.filename)
        image_path = os.path.join('temp', uuid.uuid4().hex+'.'+file.filename.strip().split(".")[-1].strip())
        
        with open(image_path, mode="wb") as file_on_disk:
            file_contents = await file.read()
            file_on_disk.write(file_contents)
            print("Image path:", image_path)
            suffix = image_path.split('.')[-1].strip().lower()
            if suffix not in ['png', 'jpg', 'jpeg']:
                return {"Error": 'Upload only images with extensions JPG, PNG or JPEG'}
        response = agent.invoke({'user_question': user_question, 'image_path': image_path, 'human_input': ''})
        print("Response: ", response['output'])
        os.remove(image_path)
        return response['output']
    except:
        print("Image is not uploaded")
        image_path = ''
        response = agent.invoke({'user_question': user_question, 'image_path': image_path, 'human_input': ''})
        print("Response: ", response['output'])
        return response['output']
        
 
    
