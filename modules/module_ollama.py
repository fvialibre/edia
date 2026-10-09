import requests
import base64

class ModelWrapper:
    def __init__(self, token, model):
        self.token = token
        self.model = model

    def chat_with_model(self, system_prompt_or_messages, user_prompt=None, base64_image=None, base64_image_2=None, temperature=1):
        url = 'https://chat.ccad.unc.edu.ar/api/chat/completions'
        headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }

        if isinstance(system_prompt_or_messages, list):
            messages = system_prompt_or_messages
        else:
            user_content = []
            if base64_image:
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })
            if base64_image_2:
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image_2}"
                    }
                })
            user_content.append({"type": "text", "text": user_prompt or ""})
            messages = [
                {
                    "role": "system",
                    "content": system_prompt_or_messages
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ]

        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature
        }
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        resp_json = response.json()
        if "choices" in resp_json and len(resp_json["choices"]) > 0:
            resp_json["content"] = resp_json["choices"][0]["message"]["content"]
        else:
            resp_json["content"] = ""
        return resp_json

    def invoke(self, system_prompt_or_messages, user_prompt=None, base64_image=None, base64_image_2=None, temperature=1):
        return self.chat_with_model(system_prompt_or_messages, user_prompt, base64_image, base64_image_2, temperature)