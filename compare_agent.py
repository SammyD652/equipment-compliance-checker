import openai
import base64

def image_to_base64(img_bytes):
    return base64.b64encode(img_bytes.read()).decode('utf-8')

def get_gpt_vision_comparison(api_key, submittal_text, image_base64):
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a QAQC engineer comparing submittal specs vs equipment nameplate details."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"""Compare the equipment submittal below to the nameplate shown in the image. 
Return a markdown table showing Field, Submittal Value, Nameplate Value, and Match (✅ or ❌).

Submittal:
{submittal_text}

Only return the table. Do not explain anything else."""},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]
    )
    return response.choices[0].message.content
