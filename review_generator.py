from openai import OpenAI
from os import getenv
from pathlib import Path
import logging
from typing import Optional, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "gpt-4-turbo"
MAX_TOKENS = 500
TEMPERATURE = 0.0
OUTPUT_FILE = "reviews_summary.txt"

def read_and_clean_reviews(file_path: str) -> Optional[List[str]]:
    try:
        reviews = Path(file_path).read_text(encoding='utf-8').splitlines()
        return [r.strip() for r in reviews if r.strip()]
    except Exception as e:
        logger.error(f"Error reading reviews file: {e}")
        return None

def generate_analysis_prompt(reviews: List[str]) -> str:
    review_text = "\n- ".join(reviews)
    return f"""Analyze reviews and respond STRICTLY in this format:
        KuKu Score: [0-100]
        Positive:
        - [One sentence summarizing positive feedback in your own words, max 30 words]
        Negative:
        - [One sentence summarizing negative feedback in your own words, max 30 words]

        Rules:
        - Base score on positive/negative ratio
        - Summarize in your own words, do not copy review text
        - Combine all positive feedback into one sentence, all negative into one
        - Keep each sentence concise (max 30 words)
        - Maintain exact formatting
        - If no negatives exist, leave the line blank

        Reviews:
        - {review_text}"""

def analyze_reviews(file_path: str) -> Optional[str]:
    client = OpenAI(api_key=getenv("OPENAI_API_KEY"))
    
    if not getenv("OPENAI_API_KEY"):
        logger.error("Missing OpenAI API key in environment variables")
        return None
    
    reviews = read_and_clean_reviews(file_path)
    if not reviews:
        logger.error("No valid reviews found or error reading file")
        return None
    
    try:
        response = client.chat.completions.create(
            model = MODEL_NAME,
            messages = [{
                "role": "system",
                "content": "You are a precise analysis engine. Follow instructions exactly."
            }, {
                "role": "user",
                "content": generate_analysis_prompt(reviews)
            }],
            temperature = TEMPERATURE,
            max_tokens = MAX_TOKENS
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"API request failed: {e}")
        return None

if __name__ == "__main__":
    result = analyze_reviews("reviews.txt")
    if result:
        try:
            with open(OUTPUT_FILE, 'w', encoding = 'utf-8') as f:
                f.write(result)
            logger.info(f"Analysis successfully written to {OUTPUT_FILE}")
        except Exception as e:
            logger.error(f"Failed to write to {OUTPUT_FILE}: {e}")
            print(f"Failed to write summary to {OUTPUT_FILE}. Check logs for details.")
    else:
        logger.error("Analysis generation failed")
        print("Failed to generate analysis. Check logs for details.")