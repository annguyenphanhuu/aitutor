import os
import base64
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

image_path = r"data\exams\images\de7plus_de01\q02.png"
base64_image = encode_image(image_path)

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

payload = {
    "model": "gpt-4o",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Please look at the graph again. Let's be extremely precise.\n1) Is the dashed horizontal line representing the asymptote ABOVE or BELOW the x-axis? By how many grid units?\n2) Look at the numbers on the vertical axis (y-axis). What is the number next to the dashed horizontal line?\n3) Look at the number above the origin on the y-axis. What number is it?\n4) If the dashed line is at y=-1, why did you previously say it was at y=1? Is there another dashed line, or is the negative sign small, or did the model confuse top/bottom?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ],
    "max_tokens": 500
}

response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
print(response.json()["choices"][0]["message"]["content"])
