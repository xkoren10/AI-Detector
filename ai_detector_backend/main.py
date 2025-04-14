from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
import xgboost as xgb
from pydantic import BaseModel
import torch
from transformers import DistilBertForSequenceClassification, AutoTokenizer, TextClassificationPipeline
import shap, os
import warnings
from fastapi.staticfiles import StaticFiles
from ai_detector_backend.xgbc.xgbc_functions import preprocess_and_vectorize
from ai_detector_backend.xgbc.xgbc_shap import save_shap_plots_to_html


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

checkpoint = "tommyliphys/ai-detector-distilbert"
model = DistilBertForSequenceClassification.from_pretrained(checkpoint, from_tf=True)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


model.config.label2id = {"human": 0, "AI": 1}
model.config.id2label = {0: "human", 1: "AI"}

pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, top_k=None, device=device)

xgbc_model = xgb.XGBClassifier()
xgbc_model.load_model("ai_detector_backend/xgbc/xgbc_model_balanced_13_feats.json")


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
        # Example usage
        try:
            processed_input = preprocess_and_vectorize(text)

            # Now predict probabilities
            class_probabilities = xgbc_model.predict_proba(processed_input)[0]  # (num_samples, num_classes)

            response = {"prediction": [[{"label": "human",
                                         "score": float(class_probabilities[1])
                                         },
                                        {"label": "AI",
                                         "score": float(class_probabilities[0])
                                         }]]}
            if explain:
                html_file = save_shap_plots_to_html(xgbc_model, processed_input)
                response["shap_explanation"] = "/static/explanation.html"
            return response

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    elif classifier == "bert":
        try:
            # Get predictions
            prediction = pipe([text])
            response = {"prediction": prediction}

            if explain:
                # Generate SHAP explanations
                explainer = shap.Explainer(pipe)
                shap_values = explainer([text])

                file_name = "explanation.html"
                file_path = os.path.join(STATIC_DIR, file_name)
                file = open(file_path, 'w')
                file.write(shap.plots.text(shap_values, display=False))
                file.close()

                response["shap_explanation"] = f"/static/{file_name}"

            return response
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    else:
        raise HTTPException(status_code=500, detail=f"Invalid classifier: {classifier}. Classifier must be \"xgbc\" or \"bert\".")
