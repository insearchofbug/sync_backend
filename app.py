from urllib import response
from fastapi import FastAPI
from scrapper.main import crawl

app = FastAPI()

@app.post("/web-crawler")
def read_root(body: dict):
    response = None
    try:
        response = crawl(body['url'])
    except:
        print("Error while doing the scrapping for ", body["url"])
    return {"message": response}
    
