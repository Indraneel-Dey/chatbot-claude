import boto3
import botocore
from botocore.config import Config

from langchain_core.output_parsers import StrOutputParser
from langchain_aws import ChatBedrock 

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from enum import Enum
from typing import Optional

boto3_bedrock = boto3.client('bedrock-runtime')

retry_config = Config(
    region_name='us-east-1',
    retries={
        'max_attempts': 10,
        'mode': 'standard'
    }
)

model_id = "anthropic.claude-3-haiku-20240307-v1:0"
sonnet_model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

model_kwargs = {
    "max_tokens": 2048,
    "temperature": 0.0,
    "top_k": 250,
    "top_p": 1,
    "stop_sequences": ["\n\nHuman"],
}

model = ChatBedrock(
    client=boto3_bedrock,
    model_id=model_id,
    model_kwargs=model_kwargs,
)

sonnet_model = ChatBedrock(
    client=boto3_bedrock,
    model_id=sonnet_model_id,
    model_kwargs=model_kwargs,
)

app = FastAPI()
handler = Mangum(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelType(str, Enum):
    SONNET = "sonnet"
    CLAUDE = "haiku"

@app.post("/query")
def query(question: str, model_type: Optional[ModelType] = None):
    if not model_type:
        model_type = ModelType.SONNET

    if model_type == ModelType.SONNET:
        result = sonnet_model.invoke(question)
    elif model_type == ModelType.CLAUDE:
        result = model.invoke(question)
    else:
        return {"error": "Invalid model type."}

    return {"result": result}

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)