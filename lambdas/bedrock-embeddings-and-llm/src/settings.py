import cfnresponse
import json

# Default prompt templates
AMAZON_GENERATE_QUERY_PROMPT_TEMPLATE = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.<br>Chat History: <br>{history}<br>Follow up question: {input}<br>Standalone question:"""
AMAZON_QA_PROMPT_TEMPLATE = """<br><br>Human: You are a friendly AI assistant. Answer the question in <question> tags only based on the provided reference passages. Here are reference passages in <references> tags:<br><references><br>{context}<br></references><br>If the references contain the information needed to respond, then write a confident response in under 50 words, quoting the relevant references. <br>Otherwise, if you can make an informed guess based on the reference passages, then write a less confident response in under 50 words, stating your assumptions.<br>Finally, if the references do not have any relevant information, then respond saying \\"Sorry, I don't know\\".<br><question><br>{query}<br></question><br><br>Assistant: According to the reference passages, in under 50 words:"""
ANTHROPIC_GENERATE_QUERY_PROMPT_TEMPLATE = """<br><br>Human: Here is a chat history in <chatHistory> tags:<br><chatHistory><br>{history}<br></chatHistory><br>Human: And here is a follow up question or statement from the human in <followUpMessage> tags:<br><followUpMessage><br>{input}<br></followUpMessage><br>Human: Rephrase the follow up question or statement as a standalone question or statement that makes sense without reading the chat history.<br><br>Assistant: Here is the rephrased follow up question or statement:"""
ANTHROPIC_QA_PROMPT_TEMPLATE = AMAZON_QA_PROMPT_TEMPLATE
AI21_GENERATE_QUERY_PROMPT_TEMPATE = ANTHROPIC_GENERATE_QUERY_PROMPT_TEMPLATE
AI21_QA_PROMPT_TEMPLATE = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know. Documents: {context} Instruction: Based on the above documents, provide a detailed answer for {query} Answer "don't know" if not present in the document. Solution:"""
COHERE_GENERATE_QUERY_PROMPT_TEMPLATE = AMAZON_GENERATE_QUERY_PROMPT_TEMPLATE
COHERE_QA_PROMPT_TEMPLATE = AMAZON_QA_PROMPT_TEMPLATE
META_GENERATE_QUERY_PROMPT_TEMPLATE = AMAZON_GENERATE_QUERY_PROMPT_TEMPLATE
META_QA_PROMPT_TEMPLATE = AMAZON_QA_PROMPT_TEMPLATE

def getEmbeddingSettings(modelId):
    provider = modelId.split(".")[0]
    settings = {}
    # Currently, only Amazon embeddings are supported
    if provider == "amazon":
        settings.update({
            "EMBEDDINGS_SCORE_THRESHOLD": 0.8,
            "EMBEDDINGS_SCORE_ANSWER_THRESHOLD": 0.6,
            "EMBEDDINGS_TEXT_PASSAGE_SCORE_THRESHOLD": 0.7,
            "EMBEDDINGS_DIMENSIONS": 1536
        })
    else:
        raise Exception("Unsupported provider for embeddings: ", provider)    
    return settings

def getModelSettings(modelId):
    params = {
        "modelId": modelId,
        "temperature": 0
    }
    params_qa = params.copy()
    # claude-3 message API params are slightly different
    provider = modelId.split(".")[0]
    if provider == "anthropic":
        if modelId.startswith("anthropic.claude-3"):
            params = {
                "model": modelId,
                "temperature": 0,
                "max_tokens": 256,
                "top_p": 1
            }
            # add optional system prompt to qa params
            params_qa = {
                "model": modelId,
                "temperature": 0,
                "max_tokens": 256,
                "top_p": 1,
                "system": "You are a helpful AI assistant."
            }
    lambdahook_args = {"Prefix":"LLM Answer:", "Model_params": params}
    settings = {
        'LLM_GENERATE_QUERY_MODEL_PARAMS': json.dumps(params),
        'LLM_QA_MODEL_PARAMS': json.dumps(params_qa),
        'QNAITEM_LAMBDAHOOK_ARGS': json.dumps(lambdahook_args)
    }
    provider = modelId.split(".")[0]
    if provider == "anthropic":
        settings.update({
        'LLM_GENERATE_QUERY_PROMPT_TEMPLATE': ANTHROPIC_GENERATE_QUERY_PROMPT_TEMPLATE,
        'LLM_QA_PROMPT_TEMPLATE': ANTHROPIC_QA_PROMPT_TEMPLATE
        })
    elif provider == "ai21":
        settings.update({
        'LLM_GENERATE_QUERY_PROMPT_TEMPLATE': AI21_GENERATE_QUERY_PROMPT_TEMPATE,
        'LLM_QA_PROMPT_TEMPLATE': AI21_QA_PROMPT_TEMPLATE
        })
    elif provider == "amazon":
        settings.update({
        'LLM_GENERATE_QUERY_PROMPT_TEMPLATE': AMAZON_GENERATE_QUERY_PROMPT_TEMPLATE,
        'LLM_QA_PROMPT_TEMPLATE': AMAZON_QA_PROMPT_TEMPLATE
        })
    elif provider == "cohere":
        settings.update({
        'LLM_GENERATE_QUERY_PROMPT_TEMPLATE': COHERE_GENERATE_QUERY_PROMPT_TEMPLATE,
        'LLM_QA_PROMPT_TEMPLATE': COHERE_QA_PROMPT_TEMPLATE
        })
    elif provider == "meta":
        settings.update({
        'LLM_GENERATE_QUERY_PROMPT_TEMPLATE': META_GENERATE_QUERY_PROMPT_TEMPLATE,
        'LLM_QA_PROMPT_TEMPLATE': META_QA_PROMPT_TEMPLATE
        })
    else:
        raise Exception("Unsupported provider: ", provider)
    return settings

def lambda_handler(event, context): 
    print("Event: ", json.dumps(event))
    status = cfnresponse.SUCCESS
    responseData = {}
    reason = ""
    if event['RequestType'] != 'Delete':
        try:                   
            llmModelId = event['ResourceProperties'].get('LLMModelId', '')
            embeddingsModelId = event['ResourceProperties'].get('EmbeddingsModelId', '')
            responseData = getModelSettings(llmModelId)
            responseData.update(getEmbeddingSettings(embeddingsModelId))
        except Exception as e:
            print(e)
            status = cfnresponse.FAILED
            reason = f"Exception thrown: {e}"              
    cfnresponse.send(event, context, status, responseData, reason=reason) 