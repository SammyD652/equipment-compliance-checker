from openai import OpenAI

def get_gpt_vision_comparison(api_key, submittal_text, image_b64):
    client = OpenAI(api_key=api_key)
    
    system_msg = {
        "role": "system",
        "content": "You are an MEP QAQC Engineer. Compare the equipment nameplate data with the technical submittal details. Output a comparison table in Markdown with fields, submittal values, image values, and match yes/no."
    }
    user_msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": f"Compare this submittal:
{submittal_text}

with this equipment nameplate image."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
        ]
    }
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[system_msg, user_msg]
    )
    
    return response.choices[0].message.content
