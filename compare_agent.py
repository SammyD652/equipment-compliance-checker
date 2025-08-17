import base64
from openai import OpenAI

def image_to_base64(uploaded_image):
    return base64.b64encode(uploaded_image.read()).decode("utf-8")

def get_gpt_vision_comparison(api_key, submittal_text, image_base64):
    client = OpenAI(api_key=api_key)

    prompt = f"""
You are a QAQC Engineer. Based on the technical submittal information below, compare it with the image of the equipment nameplate provided (in base64 format). Return a table of any mismatches or confirm if everything matches.

Respond in markdown table format with columns: Field, Submittal Value, Equipment Value, Match (✅ / ❌).

Technical Submittal:
{submittal_text}

Equipment Nameplate Image (base64):
{image_base64}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a technical QAQC assistant skilled at comparing specifications."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content
