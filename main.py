from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pandas as pd
import pickle
import PyPDF2 as pdf
from purifytext import clean_text
import numpy as np

app = FastAPI()

# Load pre-trained model and label encoder
with open("model.pkl", "rb") as model_file:
    pipeline_lr = pickle.load(model_file)

with open("label_encoder.pkl", "rb") as encoder_file:
    label_encoder = pickle.load(encoder_file)

templates = Jinja2Templates(directory="templates")


def input_pdf_text(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page_number in range(len(reader.pages)):
        page = reader.pages[page_number]
        text += str(page.extract_text())
    return text


def prediction(resume):
    resume_df = pd.DataFrame([resume], columns=["Resume"])
    clean_df = clean_text(dataframe=resume_df, column_name="Resume")
    CORPUS = np.array(clean_df["Resume"])
    pred = pipeline_lr.predict(CORPUS)
    return label_encoder.inverse_transform(pred)[0]


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/evaluate/", response_class=HTMLResponse)
async def evaluate_resume(request: Request, file: UploadFile = File(...)):
    resume_text = input_pdf_text(file.file)
    pred = prediction(resume_text)
    return templates.TemplateResponse(
        "index.html", {"request": request, "class_name": pred}
    )
