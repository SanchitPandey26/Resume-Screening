import os
from google import genai
from google.genai import types
import json

def call_llm_fit_scorer(resume_jsons, jd_json, api_key):
    client = genai.Client(api_key=api_key)
    model = "gemini-2.0-flash"

    prompt_text = (
        "For each candidate resume below, compare it to the provided job description.\n"
        "For each, return a JSON object with:\n\n"
        "name (from the resume)\n"
        "email (from the resume)\n"
        "phone (from the resume)\n"
        "fit_score (number from 0 to 10, where 10 is a perfect fit)\n"
        "matched_skills (list of skills present in both resume and JD)\n"
        "missing_skills (list of required JD skills not found in the resume)\n"
        "evaluation (a short, objective summary of the candidate’s fit, mentioning strengths and gaps)\n\n"
        "When comparing skills:\n"
        "- Focus the fit score and evaluation primarily on technical and role-specific skills.\n"
        "- Do not penalize a candidate for missing foundational technical skills (such as \"object-oriented programming\", \"data structures\", \"algorithms\") or soft skills (such as \"communication skills\", \"analytical skills\", \"teamwork\") if their education, job titles, or work experience clearly imply these skills.\n"
        "- Infer such skills from relevant job titles, degrees, leadership, or collaborative work.\n"
        "- Only list these as missing if there is clear evidence the candidate lacks them or if their experience is too junior to reasonably assume them.\n\n"
        "Return a list of such JSON objects, one per resume.\n\n"
        "Resumes:\n"
        f"{json.dumps(resume_jsons, ensure_ascii=False)}\n\n"
        "Job Description:\n"
        f"{json.dumps(jd_json, ensure_ascii=False)}"
    )

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt_text),
            ],
        )
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=0.2,
        response_mime_type="application/json",
        system_instruction=[
            types.Part.from_text(text=(
                "You are an expert HR assistant and candidate evaluator.\n"
                "Your task is to compare multiple candidate resumes to a structured job description for a software engineering role.\n"
                "Always provide your output as a list of valid minified JSON objects, one for each resume, following the requested schema.\n"
                "Be thorough, accurate, and unbiased.\n"
                "If a field is missing, return its value as null or an empty array as appropriate.\n"
                "Do not include any commentary, only the JSON."
            )),
        ],
    )

    response_text = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        response_text += chunk.text

    if response_text.strip().startswith("```"):
        response_text = response_text.strip()[7:]
    if response_text.strip().endswith("```"):
        response_text = response_text.strip()[:-3]
    return response_text
