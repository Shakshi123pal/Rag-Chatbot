import requests
import json

def chat_with_model(user_input):
    url = "http://127.0.0.1:11434/api/chat"
    data = {
        "model": "llama3.2:3b-instruct-q4_0",
        "messages": [{"role": "user", "content": user_input}]
    }

    response = requests.post(url, json=data, stream=True)
    
    full_text = ""
    for line in response.iter_lines():
        if line:
            obj = json.loads(line)
            content = obj.get("message", {}).get("content", "")
            full_text += content
            if obj.get("done", False):
                break

    return full_text

# Example usage
answer = chat_with_model("Hello! How are you?")
print(answer)
