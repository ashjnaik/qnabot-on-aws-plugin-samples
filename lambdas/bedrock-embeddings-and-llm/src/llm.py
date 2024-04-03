import boto3
import json
import os

# Defaults
DEFAULT_MODEL_ID = os.environ.get("DEFAULT_MODEL_ID","anthropic.claude-instant-v1")
AWS_REGION = os.environ["AWS_REGION_OVERRIDE"] if "AWS_REGION_OVERRIDE" in os.environ else os.environ["AWS_REGION"]
ENDPOINT_URL = os.environ.get("ENDPOINT_URL", f'https://bedrock-runtime.{AWS_REGION}.amazonaws.com')
DEFAULT_MAX_TOKENS = 256
STREAMING_ENABLED = os.environ["STREAMING_ENABLED"]
accept = "application/json"
contentType = "application/json"

# global variables - avoid creating a new client for every request
bedrock_client = None
dynamodb_client = boto3.resource('dynamodb')

def get_bedrock_client():
    print("Connecting to Bedrock Service: ", ENDPOINT_URL)
    client = boto3.client(service_name='bedrock-runtime', region_name=AWS_REGION, endpoint_url=ENDPOINT_URL)
    return client

def get_request_body(modelId, parameters, prompt):
    provider = modelId.split(".")[0]
    print('provider is : ' ,provider)
    request_body = None
    if provider == "anthropic":
        # claude-3 models use new messages format
        if modelId.startswith("anthropic.claude-3"):
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [{"role": "user", "content": [{'type':'text','text': prompt}]}],
                "max_tokens": DEFAULT_MAX_TOKENS
            }
            request_body.update(parameters)
        else:
            request_body = {
                "prompt": prompt,
                "max_tokens_to_sample": DEFAULT_MAX_TOKENS
            } 
            request_body.update(parameters)
    elif provider == "ai21":
        request_body = {
            "prompt": prompt,
            "maxTokens": DEFAULT_MAX_TOKENS
        }
        request_body.update(parameters)
    elif provider == "amazon":
        textGenerationConfig = {
            "maxTokenCount": DEFAULT_MAX_TOKENS
        }
        textGenerationConfig.update(parameters)
        request_body = {
            "inputText": prompt,
            "textGenerationConfig": textGenerationConfig
        }
    elif provider == "cohere":
        request_body = {
            "prompt": prompt,
            "max_tokens": DEFAULT_MAX_TOKENS
        }
        request_body.update(parameters)
    elif provider == "meta":
        request_body = {
            "prompt": prompt,
            "max_gen_len": DEFAULT_MAX_TOKENS
        }
        request_body.update(parameters)
    else:
        raise Exception("Unsupported provider: ", provider)
    return request_body

def close(intent_request, fulfillment_state, message):
    intent_request['sessionState']['intent']['state'] = fulfillment_state

    return {
        'sessionState': {
            'dialogAction': {
                'type': 'Close'
            },
            'intent': intent_request['sessionState']['intent']
        },
        'messages': [message],
        'sessionId': intent_request['sessionId'],
        'requestAttributes': intent_request['requestAttributes'] if 'requestAttributes' in intent_request else None
    }

def get_session_attributes(intent_request):
    print(intent_request)
    sessionState = intent_request['sessionState']
    if 'sessionAttributes' in sessionState:
        print('Session Attributes', sessionState['sessionAttributes'])
        return sessionState['sessionAttributes']

    return {}

def get_generate_text(modelId, response):
    provider = modelId.split(".")[0]
    generated_text = None
    response_body = json.loads(response.get("body").read())
    print("Response body: ", json.dumps(response_body))
    if provider == "anthropic":
        # claude-3 models use new messages format
        if modelId.startswith("anthropic.claude-3"):
            generated_text = response_body.get("content")[0].get("text")
        else:
            generated_text = response_body.get("completion")
    elif provider == "ai21":
        generated_text = response_body.get("completions")[0].get("data").get("text")
    elif provider == "amazon":
        generated_text = response_body.get("results")[0].get("outputText")
    elif provider == "cohere":
        generated_text = response_body.get("generations")[0].get("text")
    elif provider == "meta":
        generated_text = response_body.get("generation")
    else:
        raise Exception("Unsupported provider: ", provider)
    return generated_text

def call_llm(parameters, prompt, event):
    global bedrock_client

    sessionId = event['sessionId']
    sessionAttributes = get_session_attributes(event)

    modelId = parameters.pop("modelId", DEFAULT_MODEL_ID)
    body = get_request_body(modelId, parameters, prompt)
    print("ModelId", modelId, "-  Body: ", body)
    
    if (bedrock_client is None):
        bedrock_client = get_bedrock_client()
    
    fullreply = '';

    if STREAMING_ENABLED and ( 'streamingDynamoDbTable' in sessionAttributes) and ('streamingDynamoDbTable' in sessionAttributes) :
        apigatewaymanagementapi = boto3.client(
            'apigatewaymanagementapi', 
            endpoint_url = sessionAttributes['streamingEndpoint']
        )
            
        wstable = dynamodb_client.Table(sessionAttributes['streamingDynamoDbTable'])
        db_response = wstable.get_item(Key={'sessionId': sessionId})
        print (db_response)
        connectionId = db_response['Item']['connectionId']
        print('Get ConnectionID ', connectionId)

        response = bedrock_client.invoke_model_with_response_stream(
            body=json.dumps(body), modelId=modelId, accept=accept, contentType=contentType
        )
        stream = response.get('body')

        if stream:
            for streamEvent in stream:
                chunk = streamEvent.get('chunk')
                if chunk:
                    chunk_obj = json.loads(chunk.get('bytes').decode())
                    text = chunk_obj['completion']
                    fullreply = fullreply + text
                    print("chunk text sent back to the client: ",text);
                    response = apigatewaymanagementapi.post_to_connection(
                        Data=text,
                        ConnectionId=connectionId)
                    
        message = {
            'contentType': 'CustomPayload',
            'content': fullreply
        }
        fulfillment_state = "Fulfilled"
    
        return close(event, fulfillment_state, message)
    else:
        response = bedrock_client.invoke_model(body=json.dumps(body), modelId=modelId, accept=accept, contentType=contentType)
        generated_text = get_generate_text(modelId, response)
        return generated_text


"""
Example Test Event:
{
  "prompt": "\n\nHuman:Why is the sky blue?\n\nAssistant:",
  "parameters": {
    "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
    "temperature": 0,
    "system": "You are an AI assistant that always answers in ryhming couplets"
  }
}
For supported parameters for each provider model, see Bedrock docs: https://us-east-1.console.aws.amazon.com/bedrock/home?region=us-east-1#/providers
"""
def lambda_handler(event, context):
    print("Event: ", json.dumps(event))
    print("Context: ",context)
    prompt = event["prompt"]
    
    parameters = event["parameters"] 
    generated_text = call_llm(parameters, prompt, event)
    print("Result:", json.dumps(generated_text))
    return {
        'generated_text': generated_text
    }