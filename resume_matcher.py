# Resume Matcher Module using open-source ML and LLM models

from typing import Tuple

from sentence_transformers import SentenceTransformer, util
from transformers import pipeline


class ResumeMatcher:
    """A resume matcher that scores resume relevance and suggests improvements."""

    def __init__(self,
                 embedding_model: str = "sentence-transformers/paraphrase-MiniLM-L6-v2",
                 llm_model: str = "google/flan-t5-base") -> None:
        self.embedder = SentenceTransformer(embedding_model)
        self.generator = pipeline("text2text-generation", model=llm_model)

    def score_resume(self, resume_text: str, jd_text: str) -> float:
        """Return a similarity score between resume and job description."""
        resume_emb = self.embedder.encode(resume_text, convert_to_tensor=True)
        jd_emb = self.embedder.encode(jd_text, convert_to_tensor=True)
        score = float(util.cos_sim(resume_emb, jd_emb))
        # convert cosine similarity (-1 to 1) to percentage (0-100)
        normalized = (score + 1) / 2 * 100
        return round(normalized, 2)

    def suggest_improvements(self, resume_text: str, jd_text: str) -> str:
        """Generate suggestions to improve the resume for the JD."""
        prompt = (
            "You are an expert career coach. "
            "Given the following resume and job description, provide concise bullet "
            "point suggestions to improve the resume so it matches the job.\n\n"
            f"RESUME:\n{resume_text}\n\nJOB DESCRIPTION:\n{jd_text}\n\nSuggestions:")
        result = self.generator(prompt, max_length=256, num_beams=4)
        return result[0]["generated_text"].strip()

    def match(self, resume_text: str, jd_text: str) -> Tuple[float, str]:
        """Return the score and improvement suggestions."""
        score = self.score_resume(resume_text, jd_text)
        suggestions = self.suggest_improvements(resume_text, jd_text)
        return score, suggestions


class ResumeAgent:
    """Agent that iteratively improves the resume to better match the job."""

    def __init__(self, matcher: ResumeMatcher | None = None, target_score: float = 90.0, max_iter: int = 3) -> None:
        self.matcher = matcher or ResumeMatcher()
        self.target = target_score
        self.max_iter = max_iter

    def run(self, resume_text: str, jd_text: str) -> tuple[float, str, str]:
        """Return (score, improved_resume, suggestions)."""
        current_resume = resume_text
        for _ in range(self.max_iter):
            score = self.matcher.score_resume(current_resume, jd_text)
            if score >= self.target:
                break
            suggestions = self.matcher.suggest_improvements(current_resume, jd_text)
            prompt = (
                "Rewrite the resume to better match the job description based on the suggestions below.\n\n"
                f"RESUME:\n{current_resume}\n\nJOB DESCRIPTION:\n{jd_text}\n\nSUGGESTIONS:\n{suggestions}\n\nUPDATED RESUME:"
            )
            result = self.matcher.generator(prompt, max_length=512, num_beams=4)
            current_resume = result[0]["generated_text"].strip()
        final_score = self.matcher.score_resume(current_resume, jd_text)
        final_suggestions = self.matcher.suggest_improvements(current_resume, jd_text)
        return final_score, current_resume, final_suggestions


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Score a resume against a job description")
    parser.add_argument("resume", help="Path to the resume text file")
    parser.add_argument("jd", help="Path to the job description text file")
    parser.add_argument("--improve", action="store_true", help="Iteratively improve the resume")
    parser.add_argument("--iterations", type=int, default=3, help="Maximum improvement iterations")
    args = parser.parse_args()

    with open(args.resume, "r", encoding="utf-8") as r_file:
        resume_text = r_file.read()
    with open(args.jd, "r", encoding="utf-8") as jd_file:
        jd_text = jd_file.read()

    matcher = ResumeMatcher()
    if args.improve:
        agent = ResumeAgent(matcher, max_iter=args.iterations)
        score, improved, suggestions = agent.run(resume_text, jd_text)
        print(f"Final Score: {score}%")
        print("Improved Resume:\n" + improved)
        print("Suggestions:\n" + suggestions)
    else:
        score, suggestions = matcher.match(resume_text, jd_text)
        print(f"Match Score: {score}%")
        print("Suggestions:\n" + suggestions)
