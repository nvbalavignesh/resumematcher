"""Interactive resume tuning service using open-source models."""

from __future__ import annotations

import argparse
import sys

from resume_matcher import ResumeAgent, ResumeMatcher


def read_text(path: str | None, prompt: str) -> str:
    if path:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    print(prompt)
    return sys.stdin.read()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Professional ATS resume tuning service")
    parser.add_argument("--resume", help="Path to resume text file")
    parser.add_argument("--jd", help="Path to job description text file")
    parser.add_argument(
        "--iterations", type=int, default=3,
        help="Number of improvement iterations")
    args = parser.parse_args()

    resume_text = read_text(
        args.resume,
        "Paste resume text and press Ctrl-D when done:")
    jd_text = read_text(
        args.jd,
        "Paste job description text and press Ctrl-D when done:")

    matcher = ResumeMatcher()
    agent = ResumeAgent(matcher, max_iter=args.iterations)
    score, improved, suggestions = agent.run(resume_text, jd_text)
    print(f"Score after tuning: {score}%")
    print("\nFine tuned resume:\n" + improved)
    print("\nSuggestions:\n" + suggestions)


if __name__ == "__main__":
    main()
