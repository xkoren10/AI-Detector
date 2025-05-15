import shap
import matplotlib.pyplot as plt
import numpy as np
import os
import joblib
from pathlib import Path

def save_shap_plots_to_html(xgbc_model, processed_input):
    project_root = Path(__file__).resolve().parent.parent.parent
    output_dir = project_root / "static_files"
    output_dir.mkdir(parents=True, exist_ok=True)

    vectorizer = joblib.load("ai_detector_backend/xgbc/tfidf_vectorizer.pkl")
    processed_input = processed_input.tocsr()

    tfidf_feature_names = vectorizer.get_feature_names_out()
    num_tfidf = len(tfidf_feature_names)
    num_handcrafted = processed_input.shape[1] - num_tfidf

    explainer = shap.Explainer(xgbc_model)
    shap_values = explainer(processed_input)

    # --- TF-IDF Features ---
    text_shap_values = shap_values.values[0][:num_tfidf]
    text_tfidf_vector = processed_input[0, :num_tfidf]  # sparse row
    non_zero_indices = text_tfidf_vector.nonzero()[1]  # non-zero column indices

    if len(non_zero_indices) > 0:
        non_zero_ngrams = [tfidf_feature_names[i] for i in non_zero_indices]
        non_zero_shap_vals = text_shap_values[non_zero_indices]
        non_zero_tfidf_vals = text_tfidf_vector.toarray().flatten()[non_zero_indices]

        text_explanation = shap.Explanation(
            values=non_zero_shap_vals,
            base_values=shap_values.base_values[0],
            data=non_zero_tfidf_vals,
            feature_names=non_zero_ngrams
        )

        text_image_path = os.path.join(output_dir, "shap_text_plot.png")
        fig1 = plt.figure()
        shap.plots.bar(text_explanation, max_display=15, show=False)
        plt.savefig(text_image_path, format="png", bbox_inches="tight")
        plt.close(fig1)

    # --- Handcrafted Features ---
    handcrafted_shap_values = shap_values.values[0][num_tfidf:]
    handcrafted_data = processed_input[0, num_tfidf:].toarray().flatten()

    handcrafted_feature_names = [
        "text_length", "word_count", "flesch_reading_ease", "sentence_count", "syllable_count",
        "avg_word_length", "perplexity", "char_count", "unique_word_count", "unique_word_ratio",
        "stopword_count", "burstiness", "entropy"
    ]

    numeric_explanation = shap.Explanation(
        values=handcrafted_shap_values,
        base_values=shap_values.base_values[0],
        data=handcrafted_data,
        feature_names=handcrafted_feature_names
    )

    numeric_image_path = os.path.join(output_dir, "shap_numeric_plot.png")
    fig2 = plt.figure()
    shap.plots.bar(numeric_explanation, max_display=13, show=False)
    plt.savefig(numeric_image_path, format="png", bbox_inches="tight")
    plt.close(fig2)

    # --- HTML ---
    html_path = os.path.join(output_dir, "xgbc_explanation.html")
    html_content = f"""
    <html>
    <body>
        <h2>SHAP Feature Importance (Text Features)</h2>
        <img src="shap_text_plot.png" alt="Text SHAP Plot" style="max-width: 100%; height: auto;" />

        <h2>SHAP Feature Importance (Numeric Features)</h2>
        <img src="shap_numeric_plot.png" alt="Numeric SHAP Plot" style="max-width: 100%; height: auto;" />
    </body>
    </html>
    """

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return html_path
