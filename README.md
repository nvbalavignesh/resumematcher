# Resume Matcher

This module scores how well a resume matches a given job description and suggests improvements.

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the matcher from the command line by providing paths to the resume and job description text files:

```bash
python resume_matcher.py resume.txt jd.txt
```

Sample resume and job description files (`sample_resume.txt` and `sample_jd.txt`)
are included for quick testing:

```bash
python resume_matcher.py sample_resume.txt sample_jd.txt
```

The output displays a match score (0-100%) and improvement suggestions generated using open-source models.

### Agentic improvement

For an iterative approach that attempts to rewrite the resume to reach a higher match score, add the `--improve` flag:

```bash
python resume_matcher.py sample_resume.txt sample_jd.txt --improve --iterations 3
```

This runs a small agent loop that refines the resume up to the specified number of iterations and prints the final score, updated resume text and suggestions.
