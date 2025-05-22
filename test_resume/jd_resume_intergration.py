import os
from dotenv import load_dotenv
import json
from Resume_Screening.resume_model.text_extract import extract_text_and_links
from Resume_Screening.resume_model.resume_api_integration import call_gemini_api
from Resume_Screening.resume_model.jd_api_integration import call_jd_gemini_api
from Resume_Screening.resume_model.embedding_matching import ResumeJDMatcher

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

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

def main(resume_files, jd_file, top_k=1):
    matcher = ResumeJDMatcher(api_key)
    resumes_json = process_resumes(resume_files, api_key)
    jd_json = process_jd(jd_file, api_key)
    jd_json, top_resumes = matcher.semantic_match(jd_json, resumes_json, top_k=top_k)
    return jd_json, [r[1] for r in top_resumes]  # Return only the resume JSONs

if __name__ == '__main__':
    resume_files = ['../resume_data/Resume.pdf', '../resume_data/Yash_Resume.pdf', '../resume_data/sample_resume_1.pdf',
                    '../resume_data/sample_resume_2.pdf', '../resume_data/sample_resume_3.pdf', '../resume_data/sample_resume_4.pdf',
                    '../resume_data/sample_resume_5.pdf', '../resume_data/sample_resume_6.pdf', '../resume_data/sample_resume_7.pdf',
                    '../resume_data/sample_resume_8.pdf', '../resume_data/sample_resume_9.pdf', '../resume_data/sample_resume_10.pdf']
    jd_file = '../resume_data/JD.txt'
    jd_json, top_resumes_json = main(resume_files, jd_file, top_k=4)
    print('Parsed JD JSON:')
    print(jd_json)
    print('\nTop matched resume JSON:')
    for resume_json in top_resumes_json:
        print(resume_json)
        print()

