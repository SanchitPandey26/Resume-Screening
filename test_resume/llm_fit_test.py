import os
import json
from dotenv import load_dotenv
from Resume_Screening.resume_model.embedding_matching import ResumeJDMatcher
from Resume_Screening.resume_model.text_extract import extract_text_and_links
from Resume_Screening.resume_model.resume_api_integration import call_gemini_api
from Resume_Screening.resume_model.jd_api_integration import call_jd_gemini_api
from Resume_Screening.resume_model.llm_fit_scorer import call_llm_fit_scorer

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

def process_resumes(resume_files, api_key):
    parsed_resumes = []
    for file in resume_files:
        text, links = extract_text_and_links(file)
        parsed_json_str = call_gemini_api(text, links, api_key)
        parsed_json = json.loads(parsed_json_str)
        parsed_resumes.append(parsed_json)
    return parsed_resumes

def process_jd(jd_file, api_key):
    with open(jd_file, 'r', encoding='utf-8') as f:
        jd_text = f.read()
    jd_json_str = call_jd_gemini_api(jd_text, api_key)
    jd_json = json.loads(jd_json_str)
    return jd_json

def main(resume_files, jd_file, top_k=4):
    matcher = ResumeJDMatcher(api_key)
    resumes_json = process_resumes(resume_files, api_key)
    jd_json = process_jd(jd_file, api_key)
    _, top_resumes = matcher.semantic_match(jd_json, resumes_json, top_k=top_k)
    top_resume_jsons = [r[1] for r in top_resumes]
    llm_result_str = call_llm_fit_scorer(top_resume_jsons, jd_json, api_key)
    try:
        llm_result = json.loads(llm_result_str)
    except Exception:
        llm_result = llm_result_str
    if isinstance(llm_result, list):
        for item in llm_result:
            print(json.dumps(item, ensure_ascii=False))
    else:
        print(llm_result)

if __name__ == '__main__':
    resume_files = ['../resume_data/Resume.pdf', '../resume_data/Yash_Resume.pdf', '../resume_data/sample_resume_1.pdf',
                    '../resume_data/sample_resume_2.pdf', '../resume_data/sample_resume_3.pdf', '../resume_data/sample_resume_4.pdf',
                    '../resume_data/sample_resume_5.pdf', '../resume_data/sample_resume_6.pdf', '../resume_data/sample_resume_7.pdf',
                    '../resume_data/sample_resume_8.pdf', '../resume_data/sample_resume_9.pdf', '../resume_data/sample_resume_10.pdf']
    jd_file = '../resume_data/JD.txt'
    main(resume_files, jd_file, top_k=4)
