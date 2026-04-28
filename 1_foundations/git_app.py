from urllib import response
from dotenv import load_dotenv
import json
from openai import OpenAI
import os
import requests
from pypdf import PdfReader
import gradio as gr
import time

load_dotenv(override=True)
git_api_key = os.getenv('GITHUB_API_KEY')

def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )

def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"Recorded" : "OK"}

def record_unknown_question(question):
    push(f"Recording {question}")
    return {"Recorded" : "OK"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in touch and provide and email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it."
            },
            "notes": {
                "type": "string",
                "description": "The additional information about the conversation that's worth recording to give context."
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Use this tool to record any question that couldn't be answered and you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": record_user_details_json}, 
{"type": "function", "function": record_unknown_question_json}]

class Me:
    def __init__(self):
        self.github = OpenAI(api_key=git_api_key, base_url="https://models.github.ai/inference") 
        self.model_name = "openai/gpt-4o-mini"
        self.name = "Rohit Agrawal"
        reader = PdfReader("me/linkedin.pdf")
        self.linkedin = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text
        with open("me/summary.txt", "r", encoding="UTF-8") as f:
            self.summary = f.read()

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool Called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool", "content": json.dumps(result), "tool_call_id": tool_call.id})
        return results

    def system_prompt(self):
        system_prompt = f"You are acting as {self.name}. You are answering questions on {self.name}'s website. \
Particularly questions related to {self.name}'s career, background, skills, roles and experience. \
Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. \
You are given a summary of {self.name}'s background and linkedin profile which you can use to give answers. \
Be professional and engaging, as if talking to a potential client or employer who came accross the website. \
If you don't know the answer of any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using record_user_details tool."
        system_prompt += f"\n\n## Summary: \n{self.summary}\n\n## Linkedin Profile: \n{self.linkedin}\n\n"
        system_prompt += f"With this context, please chat with the user, always staying in character as {self.name}."
        return system_prompt

    def chat(self, message, history):
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
        done = False

        while not done:
            try:
                response = self.github.chat.completions.create(model=self.model_name, messages=messages, tools = tools)
            except Exception as e:
                print("Rate limit hit. Waiting...")
                time.sleep(20)
            if response.choices[0].finish_reason == "tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        return response.choices[0].message.content

if __name__ == "__main__":
    me = Me()
    gr.ChatInterface(me.chat, type = "messages").launch()
