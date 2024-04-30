# This code sample uses the 'requests' library:
# http://docs.python-requests.org
import json
import hashlib
import hmac
import boto3
import requests

from botocore.exceptions import ClientError

from ssnragtotal import AnswerTicket


def load_ticket(confluence_hostname, username, api_key, request_id):
    response = requests.get(
        f"https://{confluence_hostname}/rest/servicedeskapi/request/{request_id}",
        timeout=10,
        auth=(username, api_key),
    )

    if response.status_code == 200:
        for item in response.json()["requestFieldValues"]:
            if item["fieldId"] == "summary":
                summary = item
            elif item["fieldId"] == "description":
                description = item

        return f"{summary['value']} {description['value']}"
    else:
        print(f"Failed to get data from Jira: {response.status_code}")
        print(response.text)
        return None


def post_comment(confluence_hostname, username, api_key, request_id, comment):
    response = requests.post(
        f"https://{confluence_hostname}/rest/servicedeskapi/request/{request_id}/comment",
        timeout=10,
        data=json.dumps({"body": comment, "public": False}),
        auth=(username, api_key),
        headers={"Content-Type": "application/json"},
    )

    if 200 <= response.status_code <= 299:
        return response.json()
    else:
        print(f"Failed to get data from Jira: {response.status_code}")
        print(response.text)
        return None


def get_secret():

    secret_name = "rag-keys"
    region_name = "us-east-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=region_name)

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e

    # Parse json secret string

    return json.loads(get_secret_value_response["SecretString"])


def check_signature(secret, payload, siganture):
    hash_object = hmac.new(
        secret.encode("utf-8"),
        msg=payload.encode("utf-8"),
        digestmod=hashlib.sha256,
    )
    calculated_signature = "sha256=" + hash_object.hexdigest()

    if not hmac.compare_digest(calculated_signature, siganture):
        raise ValueError("Invalid signature")


# Create a function to handle lambda requests and extract the ticket ID
def lambda_handler(event, context):

    # Extract the ticket ID from the event
    signature = event["headers"]["X-Hub-Signature"]
    payload = event["body"]
    body = json.loads(event["body"])
    ticket_id = body.get("issue", {}).get("key", None)

    if not ticket_id:
        return {
            "statusCode": 400,
            "body": json.dumps("Ticket Key not found in event"),
        }
    credentials = get_secret()

    check_signature(credentials["JIRA_WEBHOOK_SECRET"], payload, signature)

    # Load the ticket details from the Confluence API
    ticket = load_ticket(
        credentials["CONFLUENCE_HOSTNAME"],
        credentials["CONFLUENCE_USERNAME"],
        credentials["CONFLUENCE_API_KEY"],
        ticket_id,
    )

    if not ticket:
        return {
            "statusCode": 404,
            "body": json.dumps("Ticket not found"),
        }

    # Answer the ticket using the RAG model
    rag = AnswerTicket(
        openai_key=credentials["OPENAI_API_KEY"],
        runpod_key=credentials["RUNPOD_API_KEY"],
        pinecone_key=credentials["PINECONE_API_KEY"],
    )
    answer = rag(ticket)

    # Post the answer as a comment on the ticket
    post_comment(
        credentials["CONFLUENCE_HOSTNAME"],
        credentials["CONFLUENCE_USERNAME"],
        credentials["CONFLUENCE_API_KEY"],
        ticket_id,
        answer,
    )

    return {
        "statusCode": 200,
        "body": json.dumps("Ticket processed successfully"),
    }
