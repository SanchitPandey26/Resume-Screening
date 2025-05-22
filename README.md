# Resume–JD Matching Pipeline

A modern, modular AI pipeline for parsing, matching, and evaluating resumes against job descriptions using Google Gemini LLM and embeddings. The system is wrapped as a FastAPI service for easy integration, batch processing, and cloud deployment.

---

## 🚀 Features

- 📄 **Resume & JD Parsing** (PDF/DOCX & unstructured text)
- 🤖 **Semantic Matching** with Embeddings
- 🧑‍💼 **LLM Fit Scoring** (fit score, matched/missing skills, summary)
- 📦 **Batch & API Support** (multiple resumes per call)
- ☁️ **Cloud-Ready** (Google Cloud, Azure, AWS, or local)

---

## Tech Stack

- [FastAPI](https://fastapi.tiangolo.com/)
- [Google Gemini API](https://ai.google.dev/)
- [PyMuPDF](https://pymupdf.readthedocs.io/)
- [python-docx](https://python-docx.readthedocs.io/)
- [NumPy](https://numpy.org/)
- [Uvicorn](https://www.uvicorn.org/)
- [python-dotenv](https://pypi.org/project/python-dotenv/)

---

## Getting Started

### 1. Clone the Repository
```commandline
Will upload this on github and then update it.
```

### 2. Install Dependencies

```commandline
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```dotenv
GEMINI_API_KEY=your-google-gemini-api-key-here
```

### 4. Add Your Data

- Place your resumes (`.pdf` or `.docx`) and JD file (e.g., `JD.txt`) in the `resumes/` folder.

---

## Running the FastAPI Server

```commandline
uvicorn api.app:app --reload
```

- The API will be available at [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Interactive API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Using the API

### Endpoint

- **POST** `/evaluate_resumes/`

### Parameters

- `jd_file`: Upload the JD as a `.txt` file.
- `resumes`: Upload one or more resumes (`.pdf` or `.docx`).
- `top_k`: Number of top resumes to return (default: 4).

### Example Using Swagger UI

1. Go to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
2. Click "Try it out" on `/evaluate_resumes/`
3. Upload your JD and resumes, set `top_k`, and execute.

### Example Using cURL

```commandline
curl -X 'POST'
'http://127.0.0.1:8000/evaluate_resumes/'
-F 'jd_file=@resumes/JD.txt'
-F 'resumes=@resumes/resume1.pdf'
-F 'resumes=@resumes/resume2.pdf'
-F 'top_k=4'
```

---

## Project Structure

```text
Internship/
├── resume_model/
│   ├── text_extractor.py
│   ├── resume_api_integration.py
│   ├── jd_api_integration.py
│   ├── embedding_matching.py
│   └── llm_fit_scorer.py
├── test_resume/  
│   ├── jd_parser.py
│   ├── jd_resume_intergration.py
│   ├── llm_fit_test.py
│   └── resume_parser.py
├── api/
│   └── app.py         # FastAPI app
├── resumes/           # Folder for uploaded resumes and JD
├── requirements.txt
└── .env
```

---

## Output

- Returns a list of JSON objects, each with:
  - Candidate name, email, phone
  - Fit score (0–10)
  - Matched and missing skills
  - Evaluation summary

---

## Scripts
```commandline
uvicorn api.app:app --reload # Start the FastAPI server
```
---

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## License

*This project is licensed under the [MIT License](./LICENSE) © 2025 Sanchit Pandey.*

You are free to use, modify, and distribute this software with attribution.
See the LICENSE file for details.

---

## Contact

For questions or demo requests, contact [sanchit.pdy@gmail.com].

---

**Happy coding! 🚀**
