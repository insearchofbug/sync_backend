from urllib import response
from fastapi import FastAPI
from scrapper.main import crawl

app = FastAPI()

@app.post("/web-crawler")
def read_root(body: dict):
    response = None
    try:
        response = crawl(body['url'], body.get('site_map', False))
    except Exception as e:
        print("Error while doing the scrapping for ", body["url"], "| Error: ",e)
    return {"message": response}
    
