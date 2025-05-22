import numpy as np
import os

class ResumeJDMatcher:
    def __init__(self, api_key: str):
        from google import genai
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)
        self.embedding_model = "text-embedding-004"  # Free tier, recommended

    def generate_embedding(self, text: str) -> list:
        result = self.client.models.embed_content(
            model=self.embedding_model,
            contents=[text],
        )
        return result.embeddings[0].values

    def prepare_text_for_embedding(self, parsed_json: dict) -> str:
        parts = []

        skills = (
            parsed_json.get("skills")
            or parsed_json.get("required_skills")
            or []
        )
        if isinstance(skills, list):
            parts.append(" ".join(skills))

        nice_to_have = parsed_json.get("nice_to_have_skills", [])
        if isinstance(nice_to_have, list):
            parts.append(" ".join(nice_to_have))

        education = parsed_json.get("education")
        if isinstance(education, list):
            for edu in education:
                parts.append(" ".join(str(v) for v in edu.values() if v))
        elif isinstance(education, str):
            parts.append(education)

        experience = (
            parsed_json.get("experience")
            or parsed_json.get("work_experience")
        )
        if isinstance(experience, list):
            for exp in experience:
                parts.append(" ".join(str(v) for v in exp.values() if v))

        projects = parsed_json.get("projects")
        if isinstance(projects, list):
            for proj in projects:
                parts.append(" ".join(str(v) for v in proj.values() if v))

        responsibilities = parsed_json.get("responsibilities")
        if isinstance(responsibilities, list):
            parts.append(" ".join(responsibilities))

        job_title = parsed_json.get("job_title")
        if job_title:
            parts.append(job_title)

        location = parsed_json.get("location")
        if location:
            parts.append(location)

        return " ".join(parts)

    def semantic_match(self, jd_json: dict, resumes_json: list, top_k: int = 1):
        jd_text = self.prepare_text_for_embedding(jd_json)
        jd_embedding = self.generate_embedding(jd_text)

        scored_resumes = []
        for resume_json in resumes_json:
            resume_text = self.prepare_text_for_embedding(resume_json)
            resume_embedding = self.generate_embedding(resume_text)
            score = cosine_similarity(jd_embedding, resume_embedding)
            scored_resumes.append((score, resume_json))

        scored_resumes.sort(reverse=True, key=lambda x: x[0])
        top_resumes = scored_resumes[:top_k]
        return jd_json, top_resumes

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
