import os
from dotenv import load_dotenv
from Resume_Screening.resume_model.text_extract import extract_text_and_links
from Resume_Screening.resume_model.resume_api_integration import call_gemini_api

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

resume_file = "../resume_data/Resume.pdf"  # or .docx

resume_text, resume_links = extract_text_and_links(resume_file)
json_output = call_gemini_api(resume_text, resume_links, api_key)
print(json_output)
