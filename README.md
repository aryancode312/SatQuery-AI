# 🛰️ SatQuery AI
### Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis

SatQuery AI is an AI-powered application that allows users to interact with satellite and aerial images using natural language.

Instead of manually interpreting every visual feature, users can upload an image, ask questions about it, and receive AI-generated insights through a vision-language model.

## 🚀 Key Features

- **Satellite Image Analysis:** Upload and preview satellite or aerial images.
- **Visual Question Answering:** Ask natural-language questions about uploaded images.
- **AI-Powered Interpretation:** Uses a hosted Qwen Vision-Language Model through Groq API.
- **Spatial Analysis:** Explore visible spatial relationships and geographical patterns.
- **Analysis History:** Maintain locally saved analysis records.
- **Data Export:** Export analysis results in TXT and JSON formats.
- **Model Configuration:** View model settings and prompt configuration.
- **Evaluation Framework:** Evaluate AI responses using a five-criterion scoring system.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| Streamlit | Interactive web interface |
| Groq API | Hosted AI inference |
| Qwen VLM | Image-based question answering |
| JSON | Evaluation data and records |

## ⚙️ How It Works

1. Upload a satellite or aerial image.
2. Enter a question about the image.
3. The application processes the image and question.
4. The request is sent to the hosted Qwen vision-language model through Groq API.
5. The generated response is displayed in the application.
6. Save or export the analysis for future reference.

## 💻 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/SatQuery-AI.git
```

### 2. Navigate to the project directory

```bash
cd SatQuery-AI
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

Windows:
```bash
venv\Scripts\activate
```

Linux/macOS:
```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure API Key

Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

Alternatively, configure the `GROQ_API_KEY` environment variable.

**Important:** Never upload your actual API key or secrets.toml file to GitHub.

### 6. Run the application

```bash
streamlit run app.py
```

The application will be accessible locally through your browser.

## 📊 Preliminary Evaluation

The current evaluation uses five criteria to assess generated responses.

| Evaluation Criterion | Average Score |
|---|---:|
| Feature Correctness | 3.93/5 |
| Spatial Accuracy | 3.87/5 |
| Groundedness | 2.87/5 |
| Completeness | 4.27/5 |
| Uncertainty Handling | 3.47/5 |
| **Overall Average** | **3.68/5** |

These are preliminary human-scored results from 15 responses, not validated model accuracy metrics.

## 🔮 Future Improvements

- Expand the evaluation dataset.
- Improve response groundedness and uncertainty handling.
- Introduce geospatial metadata support.
- Add map-based visualization.
- Explore specialized remote-sensing vision-language models.
- Complete public cloud deployment.

## ⚠️ Limitations

SatQuery AI is designed for preliminary image exploration and educational purposes.

It does not currently implement:
- Pixel-level image segmentation.
- Multispectral analysis.
- Georeferencing or orthorectification.
- Validated object counting.
- Survey-grade measurements.
- Automated temporal change detection.

AI-generated interpretations may contain errors and should not replace professional GIS or remote-sensing workflows.

## 👨‍💻 Developer

**Aryan Singh Rathor**

B.Tech Computer Science and Engineering  
Lovely Professional University

## 📜 License

To be determined.
