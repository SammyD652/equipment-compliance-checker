from openai import OpenAI

def get_gpt_vision_comparison(api_key, submittal_text, image_base64):
    client = OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a QAQC engineer comparing equipment nameplate data to technical submittals. Highlight any mismatches."},
            {"role": "user", "content": [
                {"type": "text", "text": f"Submittal Text:\n{submittal_text}"},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{image_base64}",
                    "detail": "high"
                }}
            ]}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content
