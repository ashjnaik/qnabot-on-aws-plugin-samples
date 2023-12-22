# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import json
import os
import uuid
import boto3

AMAZONQ_APP_ID = os.environ.get("AMAZONQ_APP_ID")
AMAZONQ_REGION = os.environ.get("AMAZONQ_REGION") or os.environ["AWS_REGION"]
AMAZONQ_ENDPOINT_URL = os.environ.get("AMAZONQ_ENDPOINT_URL") or f'https://qbusiness.{AMAZONQ_REGION}.api.aws'  
print("AMAZONQ_ENDPOINT_URL:", AMAZONQ_ENDPOINT_URL)

qbusiness_client = boto3.client(
    service_name="qbusiness", 
    region_name=AMAZONQ_REGION,
    endpoint_url=AMAZONQ_ENDPOINT_URL
)

def get_amazonq_response(prompt, context, amazonq_userid, attachments):
    print(f"get_amazonq_response: prompt={prompt}, app_id={AMAZONQ_APP_ID}, context={context}")
    input = {
        "applicationId": AMAZONQ_APP_ID,
        "userMessage": prompt,
        "userId": amazonq_userid
    }
    if context:
        if context["conversationId"]:
            input["conversationId"] = context["conversationId"]
        if context["parentMessageId"]:
            input["parentMessageId"] = context["parentMessageId"]
    else:
        input["clientToken"] = str(uuid.uuid4())
    
    if attachments:
        input["attachments"] = attachments

    print("Amazon Q Input: ", input)
    try:
        resp = qbusiness_client.chat_sync(**input)
    except Exception as e:
        print("Amazon Q Exception: ", e)
        resp = {
            "systemMessage": "Amazon Q Error: " + str(e)
        }
    print("Amazon Q Response: ", json.dumps(resp))
    return resp

def get_settings_from_lambdahook_args(event):
    lambdahook_settings = {}
    lambdahook_args_list = event["res"]["result"].get("args",[])
    print("LambdaHook args: ", lambdahook_args_list)
    if len(lambdahook_args_list):
        try:
            lambdahook_settings = json.loads(lambdahook_args_list[0])
        except Exception as e:
            print(f"Failed to parse JSON:", lambdahook_args_list[0], e)
            print("..continuing")
    return lambdahook_settings

def get_user_email(event):
    isVerifiedIdentity = event["req"]["_userInfo"].get("isVerifiedIdentity")
    if not isVerifiedIdentity:
        print("User is not verified identity")
        return "Bot_user_not_verified"
    user_email = event["req"]["_userInfo"].get("Email")
    print(f"using verified bot user email as user id: {user_email}")
    return user_email

def get_args_from_lambdahook_args(event):
    parameters = {}
    lambdahook_args_list = event["res"]["result"].get("args",[])
    print("LambdaHook args: ", lambdahook_args_list)
    if len(lambdahook_args_list):
        try:
            parameters = json.loads(lambdahook_args_list[0])
        except Exception as e:
            print(f"Failed to parse JSON:", lambdahook_args_list[0], e)
            print("..continuing")
    return parameters

def getS3File(s3Path):
    if s3Path.startswith("s3://"):
        s3Path = s3Path[5:]
    s3 = boto3.resource('s3')
    bucket, key = s3Path.split("/", 1)
    obj = s3.Object(bucket, key)
    return obj.get()['Body'].read()

def getAttachments(event):
    userFilesUploaded = event["req"]["session"].get("userFilesUploaded",[])
    attachments = []
    for userFile in userFilesUploaded:
        print(f"getAttachments: userFile={userFile}")
        attachments.append({
            "data": getS3File(userFile["s3Path"]),
            "name": userFile["fileName"]
        })
    # delete userFilesUploaded from session
    event["res"]["session"].pop("userFilesUploaded",None)
    return attachments

def format_response(event, amazonq_response):
    # get settings, if any, from lambda hook args
    # e.g: {"Prefix":"<custom prefix heading>", "ShowContext": False}
    lambdahook_settings = get_settings_from_lambdahook_args(event)
    prefix = lambdahook_settings.get("Prefix","Amazon Q Answer:")
    showContextText = lambdahook_settings.get("ShowContextText",True)
    showSourceLinks = lambdahook_settings.get("ShowSourceLinks",True)
    # set plaintext, markdown, & ssml response
    if prefix in ["None", "N/A", "Empty"]:
        prefix = None
    plainttext = amazonq_response["systemMessage"]
    markdown = amazonq_response["systemMessage"]
    ssml = amazonq_response["systemMessage"]
    if prefix:
        plainttext = f"{prefix}\n\n{plainttext}"
        markdown = f"**{prefix}**\n\n{markdown}"
    if showContextText:
        contextText = ""
        for source in amazonq_response.get("sourceAttributions",[]):
            title = source.get("title","title missing")
            snippet = source.get("snippet","snippet missing")
            url = source.get("url")
            if url:
                contextText = f'{contextText}<br><a href="{url}">{title}</a>'
            else:
                contextText = f'{contextText}<br><u><b>{title}</b></u>'
            contextText = f"{contextText}<br>{snippet}\n"
        if contextText:
            markdown = f'{markdown}\n<details><summary>Context</summary><p style="white-space: pre-line;">{contextText}</p></details>'
    if showSourceLinks:
        sourceLinks = []
        for source in amazonq_response.get("sourceAttribution",[]):
            title = source.get("title","link (no title)")
            url = source.get("url")
            if url:
                sourceLinks.append(f'<a href="{url}">{title}</a>')
        if len(sourceLinks):
            markdown = f'{markdown}<br>Sources: ' + ", ".join(sourceLinks)

    # add plaintext, markdown, and ssml fields to event.res
    event["res"]["message"] = plainttext
    event["res"]["session"]["appContext"] = {
        "altMessages": {
            "markdown": markdown,
            "ssml": ssml
        }
    }
    # preserve conversation context in session
    amazonq_context = {
        "conversationId": amazonq_response.get("conversationId"),
        "parentMessageId": amazonq_response.get("systemMessageId")
    }
    event["res"]["session"]["qnabotcontext"]["amazonq_context"] = amazonq_context
    #TODO - can we determine when Amazon Q has a good answer or not?
    #For now, always assume it's a good answer.
    #QnAbot sets session attribute qnabot_gotanswer True when got_hits > 0
    event["res"]["got_hits"] = 1
    return event

def lambda_handler(event, context):
    print("Received event: %s" % json.dumps(event))
    args = get_args_from_lambdahook_args(event)
    # prompt set from args, or from req.question if not specified in args.
    userInput = args.get("Prompt", event["req"]["question"])
    qnabotcontext = event["req"]["session"].get("qnabotcontext",{})
    amazonq_context = qnabotcontext.get("amazonq_context",{})
    attachments = getAttachments(event)
    amazonq_userid = os.environ.get("AMAZONQ_USER_ID")
    if not amazonq_userid:
        amazonq_userid = get_user_email(event)
    else:
        print(f"using configured default user id: {amazonq_userid}")
    amazonq_response = get_amazonq_response(userInput, amazonq_context, amazonq_userid, attachments)
    event = format_response(event, amazonq_response)
    print("Returning response: %s" % json.dumps(event))
    return event
