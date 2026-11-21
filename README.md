# SatQuery AI
### Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis

SatQuery AI is a vision-language application designed to analyse satellite and aerial imagery through natural-language questions. It uses a vision-language model to identify visible features, describe spatial arrangements, and provide evidence-based interpretations of remote-sensing images.

The application combines Streamlit, the Groq API, and Qwen to provide an interactive image-analysis workspace with persistent analysis history and export capabilities.

## Features

- **Image Analysis:** Upload satellite or aerial images and ask natural-language questions.
- **Visual Question Answering:** Identify visible features and describe spatial relationships.
- **Evidence-Grounded Responses:** Uses a dedicated production prompt to encourage observations, cautious interpretations, and explicit uncertainty.
- **Analysis History:** Stores previous analysis records locally.
- **Export Options:** Export individual analyses and workspace results in JSON and TXT formats.
- **Evaluation Framework:** Evaluate model responses using a structured, human-reviewed evaluation dataset.
- **Settings:** Inspect the configured model and production prompt.

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| User Interface | Streamlit |
| Vision-Language Model | Qwen 3.8 27B |
| Inference Provider | Groq API |
| Image Processing | Pillow |
| Evaluation | Custom Python evaluation module |
| Data Storage | JSON |

## Project Structure

```text
satquery_ai/
├── .streamlit/
│   └── secrets.example.toml
├── data/
│   ├── images/
│   └── evaluation_dataset.json
├── prompts/
│   └── prompt_C_v3_production.txt
├── results/
│   ├── evaluation_results.json
│   └── revised_prompt_results.json
├── src/
│   ├── __init__.py
│   └── evaluation.py
├── app.py
├── record_score.py
├── requirements.txt
├── README.md
└── .gitignore
```

Local application history and API credentials are excluded from version control.

## Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd satquery_ai
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Groq API key

Create a file named `secrets.toml` inside `.streamlit/`.

Copy the example configuration:

```toml
GROQ_API_KEY = "your-groq-api-key"
```

Never commit your actual API key.

### 5. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal.

## Evaluation

SatQuery AI includes a structured evaluation framework covering five criteria:

1. Feature Correctness
2. Spatial Accuracy
3. Groundedness
4. Completeness
5. Uncertainty Handling

The evaluation dataset contains 15 image entries across five categories:

- Urban
- Agriculture
- Forest
- Water and Coastal
- Barren Land

Each image entry contains three question types: feature identification, spatial reasoning, and uncertainty assessment.

### Baseline Evaluation Results

The existing baseline contains 15 recorded responses.

| Evaluation Criterion | Average Score |
|---|---:|
| Feature Correctness | 3.93/5 |
| Spatial Accuracy | 3.87/5 |
| Groundedness | 2.87/5 |
| Completeness | 4.27/5 |
| Uncertainty Handling | 3.47/5 |
| **Overall Average** | **3.68/5** |

Scores represent reviewer judgments, not model-reported confidence.

### Evaluation Commands

List available evaluation questions:

```bash
python -m src.evaluation list
```

View evaluation results:

```bash
python -m src.evaluation summary
```

## Limitations

- Visual interpretations may contain unsupported assumptions and require human verification.
- The application does not perform pixel-level semantic segmentation.
- It does not provide verified georeferencing or geographic coordinates.
- It does not calculate quantitative land-cover percentages.
- Image interpretation depends on image quality, model capabilities, and prompt constraints.

## Future Improvements

- Improve evidence-grounded visual reasoning.
- Expand the image evaluation dataset.
- Introduce more systematic human evaluation.
- Explore remote-sensing-specific vision-language models.
- Improve robustness and deployment capabilities.

## Project Objective

The goal of SatQuery AI is to make satellite and aerial image interpretation more accessible through natural-language interaction while encouraging transparent, evidence-based visual reasoning.

---

**SatQuery AI — Interactive Vision-Language Assistant for Remote Sensing Image Analysis.**