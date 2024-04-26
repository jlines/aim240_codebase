# This code sample uses the 'requests' library:
# http://docs.python-requests.org
import requests
import json
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


with open("credentials.json") as credentials_file:
    credentials = json.loads(credentials_file.read())

ticket_id = "D6CS-320"

ticket = load_ticket(
    credentials["CONFLUENCE_HOSTNAME"],
    credentials["CONFLUENCE_USERNAME"],
    credentials["CONFLUENCE_API_KEY"],
    ticket_id,
)

rag = AnswerTicket()
answer = rag(ticket)

post_comment(
    credentials["CONFLUENCE_HOSTNAME"],
    credentials["CONFLUENCE_USERNAME"],
    credentials["CONFLUENCE_API_KEY"],
    ticket_id,
    answer,
)
