import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

def create_prompt_content(resume_text: str, resume_links: list) -> list:
    links_str = '\n'.join([f"{link['uri']}" for link in resume_links])
    prompt_text = f"""Extract the following fields from the provided resume text and return them as a single JSON object:

name

email

phone

linkedin_url

github_url

skills (list; include both explicitly listed skills and those that can be reasonably inferred from project descriptions and work experience)

education (list of objects: degree, institution, year, grade or score if available)

projects (list of objects: title, description)

work_experience (list of objects: company, role, start_date, end_date, description)
achievements (list)

When extracting skills:
- Include all professional skills, tools, technologies, programming languages, libraries, frameworks, methodologies, and techniques that are either explicitly listed or can be reasonably inferred from the candidate’s project descriptions and work experience.
- For each project or experience, infer relevant skills based on the tools, methods, or practices described, even if they are not explicitly listed in the skills section.
- Do not include domain-specific knowledge (such as “Law”, “Medicine”, “Finance”, “Education”, etc.) as a skill unless the candidate’s degree, job title, or professional background is specifically in that domain.
- Only include skills that are relevant to the candidate’s actual education, profession, or demonstrated expertise.
- Avoid listing general domain topics (like “Legal”, “Healthcare”, “Business”, “Education”) as skills unless the candidate is specialized or formally trained in that field.

For each education entry, extract the degree, institution, year, and any grade, score, or percentage if available, regardless of education level.

Do not include grades, percentages, or scores in the achievements list if they are already present in the education section. Achievements should only include awards, honors, competitions, or recognitions, not academic grades or marks.

If a field is missing, set its value to null (for strings/URLs) or an empty array (for lists).

Resume text:
{resume_text}

Resume Links:
{links_str}
"""
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt_text),
            ],
        )
    ]
    return contents

def call_gemini_api(resume_text: str, resume_links: list, api_key: str):
    client = genai.Client(api_key=api_key)
    model = "gemini-2.0-flash"
    contents = create_prompt_content(resume_text, resume_links)
    generate_content_config = types.GenerateContentConfig(
        temperature=0.2,
        response_mime_type="application/json",
        system_instruction=[
            types.Part.from_text(text="""You are an expert HR assistant and resume parsing specialist.
Extract structured candidate information from resumes for software engineering roles.
Always return output as a valid minified JSON object following the requested schema.
Be exhaustive, accurate, and unbiased.
If a field is missing, return its value as null or an empty array as appropriate.
Do not include any commentary, only the JSON."""),
        ],
    )
    response_text = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        response_text += chunk.text
    return response_text
