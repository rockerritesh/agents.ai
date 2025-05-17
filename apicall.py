from openai import OpenAI
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import numpy as np

# Load environment variables
load_dotenv()

def get_client():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("BASE_URL_OPENAI"))
    return client

client = get_client()


def get_reply(message: list, base_model: type) -> BaseModel:
    """Generate a response from OpenAI based on the conversation."""
    model_name = os.getenv("MODEL_NAME")
    # print("Model Name: ", model_name)
    try:
        completion = client.beta.chat.completions.parse(
            model= model_name,
            messages=message,
            response_format=base_model,
        )

        response = completion.choices[0].message.parsed
    except Exception as e:
        completion = client.beta.chat.completions.parse(
            model= model_name,
            messages=message,
            response_format=base_model.model_json_schema()
        )
        response = completion.choices[0].message.content
    # print(response)
    return response

def get_embedding(text: str) -> np.ndarray:
    """Generate an embedding for the given text."""
    model_name = os.getenv("EMBEDDING_MODEL_NAME")
    response = client.embeddings.create(
        model=model_name,
        input=text
    )
    return np.array(response.data[0].embedding)

# Example usage for getting a reply

class Reply(BaseModel):
    reply: str


# message = [
#     {
#         "role": "user",
#         "content": "What is the capital of France?"
#     }
# ]

# response = get_reply(message, Reply)
# print(response.reply)  # Output: Paris
# # You can also use the response as a dictionary
# print(response.dict())  # Output: {'reply': 'Paris'}

# # Example usage for getting an embedding
# text = "The capital of France is Paris."
# embedding = get_embedding(text)
# print(embedding)  # Output: Embedding vector for the text
# # You can also check the shape of the embedding
# print(embedding.shape)  # Output: (dimension,) e.g., (1536,)