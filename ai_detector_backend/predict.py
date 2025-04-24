from fastapi import HTTPException
import shap, os
from ai_detector_backend.xgbc.xgbc_functions import preprocess_and_vectorize
from ai_detector_backend.xgbc.xgbc_shap import save_shap_plots_to_html

STATIC_DIR = "static_files"

def predict_bert (pipe, text, explain:bool = False):
    try:
        # Get predictions
        prediction = pipe([text])
        response = {"prediction": prediction}

        if explain:
            # Generate SHAP explanations
            explainer = shap.Explainer(pipe)
            shap_values = explainer([text])

            file_name = "bert_explanation.html"
            file_path = os.path.join(STATIC_DIR, file_name)
            file = open(file_path, 'w')
            file.write(shap.plots.text(shap_values, display=False))
            file.close()

            response["shap_explanation"] = f"/static/{file_name}"
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



def predict_xgbc (xgbc_model, text, explain:bool = False):
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
            response["shap_explanation"] = "/static/xgbc_explanation.html"
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))