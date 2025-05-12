from fastapi import FastAPI, HTTPException
import xgboost as xgb
from pydantic import BaseModel
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
import os
import warnings
from fastapi.staticfiles import StaticFiles
from ai_detector_backend.predict import predict_bert, predict_xgbc


warnings.filterwarnings('ignore')

from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()
# Define the static directory to serve files from
STATIC_DIR = "static_files"
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to allow specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pre-trained BERT model and tokenizer

#device = 'cuda' if torch.cuda.is_available() else torch.device("cpu")  // TODO: Try to solve this
device = torch.device("cpu")
print(device)

model = AutoModelForSequenceClassification.from_pretrained('/work/ai_detector_backend/bert/models/trained_model')
model.to(device)

tokenizer = AutoTokenizer.from_pretrained('/work/ai_detector_backend/bert/models/trained_tokenizer')
model.eval()  # Set model to evaluation mode

model.config.label2id = {"human": 1, "AI": 0}   # Check this twice
model.config.id2label = {1: "human", 0: "AI"}

pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, top_k=None, device=device)

xgbc_model = xgb.XGBClassifier()
xgbc_model.load_model("/work/ai_detector_backend/xgbc/xgbc_model_balanced_13_feats.json")


class TextRequest(BaseModel):
    text: str
    classifier: str
    explain: bool


@app.post("/predict")
async def predict(request: TextRequest):
    text = request.text
    classifier = request.classifier.lower()
    explain = request.explain

    if classifier == "xgbc":
        return predict_xgbc(xgbc_model, text, explain)

    elif classifier == "bert":
        return predict_bert(pipe, text, explain)

    elif classifier == "combined":
        predictions = [predict_xgbc(xgbc_model, text, explain), predict_bert(pipe, text, explain)]

        # Weights for each prediction
        weight_xgbc = 0.6
        weight_bert = 0.4
        total_weight = weight_xgbc + weight_bert

        # Extract scores
        # Initialize label scores
        scores = {}

        # Loop through each prediction result and its associated weight
        for prediction, weight in zip(predictions, [weight_xgbc, weight_bert]):
            for item in prediction["prediction"][0]:
                label = item["label"]
                score = item["score"]

                if label not in scores:
                    scores[label] = 0.0
                scores[label] += score * weight

        # Normalize by total weight
        for label in scores:
            scores[label] /= total_weight

        # Rebuild final result
        response = {
            "prediction": [
                [
                    {"label": label, "score": scores[label]} for label in scores
                ]
            ]
        }

        if explain:
            if explain:

                xgbc_shap = "static_files/" + str(predictions[0]["shap_explanation"]).split("/")[-1]
                bert_shap = "static_files/" + str(predictions[1]["shap_explanation"]).split("/")[-1]

                with open(bert_shap, "r") as f1, open(xgbc_shap, "r") as f2:
                    content1 = f1.read()
                    content2 = f2.read()

                with open("static_files/combined_explanation.html", "w") as out:
                    out.write(content1 + "\n" + content2)
                out.close()

                response["shap_explanation"] = "/static/combined_explanation.html"

        return response
    else:
        raise HTTPException(status_code=400, detail=f"Invalid classifier: {classifier}."
                                                    f" Classifier must be \"xgbc\", \"bert\" or \"combined\".")
