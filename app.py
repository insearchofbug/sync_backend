from urllib import response
from fastapi import FastAPI
from scrapper.main import crawl
from chatbot.chat_model import ChatModel

app = FastAPI()

@app.post("/web-crawler")
def read_root(body: dict):
    response = None
    try:
        response = crawl(body['url'], body.get('site_map', False))
    except Exception as e:
        print("Error while doing the scrapping for ", body["url"], "| Error: ",e)
    return {"message": response}
    

@app.post("/chatbot-controller")
def read_chatbot_controller(body:dict):
    print(body)
    response = ""
    try:
        if body.get('action', '') == 'train':
            response = ChatModel().train_model(body['training_data_filename'])
        elif body.get('action', '') == 'context':
            response = ChatModel().ask_chatbot(body['embedding_filename'], body['question'])
    except Exception as e:
        print("chatbot controller api crash : ", e)
    
    return {"message": response}

