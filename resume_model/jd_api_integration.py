import os
from google import genai
from google.genai import types

def call_jd_gemini_api(jd_text: str, api_key: str) -> str:
    client = genai.Client(api_key=api_key)
    model = "gemini-2.0-flash"

    prompt_text = (
        "Extract the following fields from the provided job description and return them as a single JSON object:\n\n"
        "job_title\n"
        "required_skills (list)\n"
        "nice_to_have_skills (list)\n"
        "education (string; required degree(s) or field(s))\n"
        "experience_level (string; e.g., '2+ years', 'Entry Level', etc.)\n"
        "responsibilities (list)\n"
        "location (string; if available)\n\n"
        "Ignore sections about company overview, benefits, perks, application process, or any information not directly relevant to the candidate’s qualifications or job requirements.\n\n"
        "For skills, separate required skills (explicitly stated as 'required', 'must have', or essential) and nice-to-have skills (those listed as 'preferred', 'bonus', or 'optional').\n\n"
        "If a field is missing, set its value to null (for strings) or an empty array (for lists).\n\n"
        f"Job Description:\n{jd_text}"
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
                "You are an expert HR assistant and job description parsing specialist.\n"
                "Your task is to extract key hiring criteria from unstructured job descriptions for software engineering roles.\n"
                "Always provide your output as a valid minified JSON object following the requested schema.\n"
                "Be exhaustive, accurate, and unbiased.\n"
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

    return response_text
