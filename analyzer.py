"""
Resume-JD Match & Gap Analyzer (Gemini free-tier version)
------------------------------------------------------------
What this script does, step by step:
  1. Reads your resume and a job description from text files.
  2. Splits the job description into individual requirement lines (chunks).
  3. Uses Google Gemini's embedding model to turn each requirement, and your
     whole resume, into vectors (lists of numbers that capture meaning).
  4. Compares each requirement against your resume using cosine similarity
     (a standard way to measure how "close" two pieces of text are in meaning).
  5. Flags requirements below a similarity threshold as GAPS.
  6. Sends those gaps to Gemini to get a plain-English explanation + suggestion.
  7. Prints an overall match score and a gap report.

Run it with:  python analyzer.py
"""

import os
import re
import time
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors

# ---------- Setup ----------
load_dotenv()  # reads the .env file and loads GEMINI_API_KEY into the environment
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

EMBED_MODEL = "models/gemini-embedding-001"  # confirmed via list_models.py
CHAT_MODEL = "models/gemini-flash-latest"    # confirmed via list_models.py
GAP_THRESHOLD = 0.55                         # similarity below this = flagged as a gap


# ---------- Step 1: Load files ----------
def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------- Step 2: Split the JD into individual requirement lines ----------
def split_jd_into_requirements(jd_text):
    """
    Splits on newlines and bullet characters, drops empty/very short lines,
    and drops boilerplate lines (pay, location, title) that aren't actual
    technical requirements.
    """
    lines = re.split(r"[\n•]", jd_text)
    requirements = [line.strip("-• \t") for line in lines]
    requirements = [r for r in requirements if len(r) > 15]

    # Drop non-requirement boilerplate lines
    skip_patterns = [
        r"^pay\s*:", r"^salary", r"^₹", r"^work location",
        r"^location\s*:", r"^job type", r"^experience\s*:",
    ]
    requirements = [
        r for r in requirements
        if not any(re.search(p, r, re.IGNORECASE) for p in skip_patterns)
    ]
    # Drop the job title line itself (it's a heading, not a requirement)
    requirements = [r for r in requirements if not r.lower().startswith("ai developer")]

    return requirements


# ---------- Small helper: retry on transient server errors ----------
def call_with_retry(fn, max_attempts=4, base_delay=2):
    """
    Retries a function if Google's servers return a transient error
    (503 UNAVAILABLE, 429 rate limit). Waits longer between each attempt.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except genai_errors.ServerError as e:
            if attempt == max_attempts:
                raise
            wait = base_delay * attempt
            print(f"  (server busy, retrying in {wait}s... attempt {attempt}/{max_attempts})")
            time.sleep(wait)
        except genai_errors.ClientError as e:
            if "RESOURCE_EXHAUSTED" in str(e) and attempt < max_attempts:
                wait = base_delay * attempt * 2
                print(f"  (rate limited, retrying in {wait}s... attempt {attempt}/{max_attempts})")
                time.sleep(wait)
            else:
                raise


# ---------- Step 3: Get embeddings ----------
def get_embedding(text):
    response = call_with_retry(
        lambda: client.models.embed_content(model=EMBED_MODEL, contents=text)
    )
    return np.array(response.embeddings[0].values)


def get_embeddings_batch(texts):
    return [get_embedding(t) for t in texts]


# ---------- Step 4: Cosine similarity ----------
def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---------- Step 5 & 6: Match + explain gaps ----------
def explain_gap(requirement, resume_text):
    prompt = f"""You are helping a candidate understand a gap between their resume and a job requirement.

Job requirement: "{requirement}"

Candidate's resume:
\"\"\"{resume_text}\"\"\"

In 2 short sentences: (1) honestly state whether/how the resume covers this requirement,
and (2) give one concrete, realistic suggestion for closing the gap. Be direct, no fluff."""

    response = call_with_retry(
        lambda: client.models.generate_content(model=CHAT_MODEL, contents=prompt)
    )
    return response.text.strip()


# ---------- Main pipeline ----------
def main():
    print("Loading resume and job description...")
    resume_text = load_text("resume.txt")
    jd_text = load_text("jd.txt")

    print("Splitting JD into individual requirements...")
    requirements = split_jd_into_requirements(jd_text)
    print(f"  Found {len(requirements)} requirement lines.\n")

    print("Embedding resume...")
    resume_embedding = get_embedding(resume_text)

    print("Embedding requirements...")
    requirement_embeddings = get_embeddings_batch(requirements)

    print("Scoring each requirement against your resume...\n")
    results = []
    for req, emb in zip(requirements, requirement_embeddings):
        score = cosine_similarity(resume_embedding, emb)
        results.append((req, score))

    results.sort(key=lambda x: x[1], reverse=True)

    overall_score = float(np.mean([r[1] for r in results]))
    matched = [r for r in results if r[1] >= GAP_THRESHOLD]
    gaps = [r for r in results if r[1] < GAP_THRESHOLD]

    print("=" * 60)
    print(f"OVERALL MATCH SCORE: {overall_score:.2%}")
    print(f"Matched requirements: {len(matched)} / {len(results)}")
    print(f"Flagged gaps: {len(gaps)} / {len(results)}")
    print("=" * 60)

    print("\n✅ STRONGEST MATCHES:")
    for req, score in matched[:5]:
        print(f"  [{score:.2f}] {req}")

    print("\n⚠️  GAPS (below threshold):")
    for req, score in gaps:
        print(f"  [{score:.2f}] {req}")

    print("\nGenerating gap explanations (this calls GPT for each gap)...\n")
    for req, score in gaps:
        explanation = explain_gap(req, resume_text)
        print(f"- Requirement: {req}")
        print(f"  {explanation}\n")


if __name__ == "__main__":
    main()
