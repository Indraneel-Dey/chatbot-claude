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

# Create chains with the prompt, models, and output parser
# chain = prompt | model | StrOutputParser()
# sonnet_chain = prompt | sonnet_model | StrOutputParser()

# # Define the prompt messages
# messages = [
#     ("system", 
#         """
#         ### Task
#         Generate a SQL query for a RedShift Database to answer [QUESTION]{question}[/QUESTION]

#         ### Instructions
#         - If you cannot answer the question with the available database schema, return 'I do not know'
#         - Remember that revenue is price multiplied by quantity
#         - Remember that cost is supply_price multiplied by quantity

#         ### Database Schema
#         This query will run on a database whose schema is represented in this string:
#         CREATE TABLE products (
#           product_id INTEGER PRIMARY KEY, -- Unique ID for each product
#           name VARCHAR(50), -- Name of the product
#           price DECIMAL(10,2), -- Price of each unit of the product
#           quantity INTEGER  -- Current quantity in stock
#         );

#         CREATE TABLE customers (
#            customer_id INTEGER PRIMARY KEY, -- Unique ID for each customer
#            name VARCHAR(50), -- Name of the customer
#            address VARCHAR(100) -- Mailing address of the customer
#         );

#         CREATE TABLE salespeople (
#           salesperson_id INTEGER PRIMARY KEY, -- Unique ID for each salesperson
#           name VARCHAR(50), -- Name of the salesperson
#           region VARCHAR(50) -- Geographic sales region
#         );

#         CREATE TABLE sales (
#           sale_id INTEGER PRIMARY KEY, -- Unique ID for each sale
#           product_id INTEGER, -- ID of product sold
#           customer_id INTEGER,  -- ID of customer who made purchase
#           salesperson_id INTEGER, -- ID of salesperson who made the sale
#           sale_date DATE, -- Date the sale occurred
#           quantity INTEGER -- Quantity of product sold
#         );

#         CREATE TABLE product_suppliers (
#           supplier_id INTEGER PRIMARY KEY, -- Unique ID for each supplier
#           product_id INTEGER, -- Product ID supplied
#           supply_price DECIMAL(10,2) -- Unit price charged by supplier
#         );

#         -- sales.product_id can be joined with products.product_id
#         -- sales.customer_id can be joined with customers.customer_id
#         -- sales.salesperson_id can be joined with salespeople.salesperson_id
#         -- product_suppliers.product_id can be joined with products.product_id

#         ### Answer
#         Given the database schema, here is the SQL query that answers. Just return the sql query nothing else no explaination nothing.
#         """
#     ),
#     ("human", "{question}"),
# ]

# # Create a prompt template from the messages
# prompt = ChatPromptTemplate.from_messages(messages)

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