import os
import boto3
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Get configs from environment
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v1")
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "meta.llama2-70b-chat-v1")

# Bedrock boto3 client
bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
