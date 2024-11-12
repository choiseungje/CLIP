import torch
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from model import CLIP
from utils import korean_to_english
import os

# FastAPI 앱 초기화
app = FastAPI()

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), '../templates'))

# CLIP 모델과 장치 설정

device = "cuda" if torch.cuda.is_available() else "cpu"


# 이미지 데이터셋 경로 설정
IMAGE_DIR = os.path.join(os.path.dirname(__file__), '..', 'output_images')
STYLES_DIR = os.path.join(os.path.dirname(__file__), '..', 'styles')

app.mount("/static", StaticFiles(directory=IMAGE_DIR), name="static")
app.mount("/styles", StaticFiles(directory=STYLES_DIR), name="styles")
model = CLIP(device, image_dir=IMAGE_DIR)
model.load_image()


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/image/", response_class=HTMLResponse)
async def get_image(request: Request, prompt: str = Form(...)):
    try:
        prompt = korean_to_english(prompt)
        relative_image_path = model.get_best_matching_image(prompt=prompt)
        print(relative_image_path)
        return templates.TemplateResponse("result.html", {"request": request, "image_path": relative_image_path})


    except Exception as e:
        return templates.TemplateResponse("error.html", {"request": request, "error_message": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
