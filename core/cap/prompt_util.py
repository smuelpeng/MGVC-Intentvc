import os
import json
import re

ALL_CATEGORIES = [
    "airplane", "basketball", "bear", "bicycle", "bird", "boat",
    "book", "bottle", "bus", "car", "cat", "cattle",
    "chameleon", "coin", "crab", "crocodile", "cup", "deer",
    "dog", "drone", "electricfan", "elephant", "flag", "fox",
    "frog", "gametarget", "gecko", "giraffe", "goldfish", "gorilla",
    "guitar", "hand", "hat", "helmet", "hippo", "horse",
    "kangaroo", "kite", "leopard", "licenseplate", "lion", "lizard",
    "microphone", "monkey", "motorcycle", "mouse", "person", "pig",
    "pool", "rabbit", "racing", "robot", "rubicCube", "sepia",
    "shark", "sheep", "skateboard", "spider", "squirrel", "surfboard",
    "swing", "tank", "tiger", "train", "truck", "turtle",
    "umbrella", "volleyball", "yoyo", "zebra"
]


def get_core_prompt(user_content):
    """
    """
    core_prompt = []
    end_flag = False
    for line in user_content.split("\n"):
        core_prompt.append(line)
        if line.startswith("- "):
            end_flag = True
        if end_flag and not line.startswith("-"):
            break
    core_prompt = "\n".join(core_prompt)
    return core_prompt


def get_prompt(category):
    with open(f"prompts/{category}.txt", "r") as f:
        data = f.read()

    frame_list = []
    dam_caption = ""
    rag_captions = {}
    data = data.replace("{rag_captions[video_id]}", "")
    if data.startswith("messages"):
        # Handle Python variable assignment format
        # Extract the messages list from the assignment
        messages_str = data.replace("messages = ", "")
        messages = eval(messages_str)
        system_content = messages[0]["content"]
        user_content = messages[1]["content"][1]["text"]
    else:
        parts = data.split("},")

        # Extract system content (first part)
        system_part = parts[0] + "}"
        system_data = eval(system_part)
        system_content = system_data["content"]

        # print('system_content: ',system_content)
        # Extract user content (second part)
        user_part = "},".join(parts[1:])

        user_part = eval(user_part)
        user_content = user_part["content"][1]["text"]
        # print('user_content: ',user_content)
    core_prompt = get_core_prompt(user_content)

    return {
        "system_content": system_content,
        "user_content": user_content,
        "core_prompt": core_prompt
    }

class IntentPromptSchema:
    def __init__(self):
        prompt_schema_file_path = os.path.join(os.path.dirname(__file__), "prompt_schema.json")
        self.prompt_schema = json.load(open(prompt_schema_file_path, "r"))
        self.ALL_CATEGORIES = ALL_CATEGORIES
        
    def get_prompt(self, category):
        return self.prompt_schema[category]
    
    def get_all_categories(self):
        return self.ALL_CATEGORIES


if __name__ == "__main__":

    prompt_schema = {}
    for category in ALL_CATEGORIES:
        prompt_schema[category] = {
            "system_content": get_prompt(category)['system_content'],
            "core_prompt": get_prompt(category)['core_prompt']
        }
    with open("prompt_schema.json", "w") as f:
        json.dump(prompt_schema, f, indent=4)