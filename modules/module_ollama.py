import requests
import base64

class ModelWrapper:
    def __init__(self, token, model):
        self.token = token
        self.model = model

    def chat_with_model(self, system_prompt, user_prompt, base64_image=None):
        url = 'https://chat.ccad.unc.edu.ar/api/chat/completions'
        headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }

        user_content = []
        if base64_image:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
        user_content.append({"type": "text", "text": user_prompt})
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_content
            }
        ]

        data = {
            "model": self.model,
            "messages": messages,
            "temperature": 1
        }
        response = requests.post(url, headers=headers, json=data)
        response = response.json()
        response['content'] = response['choices'][0]['message']['content']
        return response

    def invoke(self, system_prompt, user_prompt, base64_image=None):
        return self.chat_with_model(system_prompt, user_prompt, base64_image)