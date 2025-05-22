from fastapi import FastAPI, UploadFile, File, Form
from typing import List
import os
import json
from dotenv import load_dotenv
from Resume_Screening.resume_model.text_extract import extract_text_and_links
from Resume_Screening.resume_model.resume_api_integration import call_gemini_api
from Resume_Screening.resume_model.jd_api_integration import call_jd_gemini_api
from Resume_Screening.resume_model.embedding_matching import ResumeJDMatcher
from Resume_Screening.resume_model.llm_fit_scorer import call_llm_fit_scorer

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

app = FastAPI()

@app.post("/evaluate_resumes/")
async def evaluate_resumes(
    jd_file: UploadFile = File(...),
    resumes: List[UploadFile] = File(...),
    top_k: int = Form(4)
):
    jd_text = (await jd_file.read()).decode("utf-8")
    jd_json_str = call_jd_gemini_api(jd_text, api_key)
    jd_json = json.loads(jd_json_str)

    parsed_resumes = []
    for resume_file in resumes:
        content = await resume_file.read()
        temp_path = f"/tmp/{resume_file.filename}"
        with open(temp_path, "wb") as f:
            f.write(content)
        text, links = extract_text_and_links(temp_path)
        parsed_json_str = call_gemini_api(text, links, api_key)
        parsed_json = json.loads(parsed_json_str)
        parsed_resumes.append(parsed_json)
        os.remove(temp_path)

    matcher = ResumeJDMatcher(api_key)
    _, top_resumes = matcher.semantic_match(jd_json, parsed_resumes, top_k=top_k)
    top_resume_jsons = [r[1] for r in top_resumes]

    llm_result_str = call_llm_fit_scorer(top_resume_jsons, jd_json, api_key)
    try:
        llm_result = json.loads(llm_result_str)
    except Exception:
        llm_result = llm_result_str

    if isinstance(llm_result, list):
        return "\n".join(json.dumps(item, ensure_ascii=False) for item in llm_result)
    else:
        return llm_result

@app.get("/")
def read_root():
    return {"message": "Resume-JD Fit Scoring API is running."}
