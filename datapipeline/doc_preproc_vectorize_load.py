import nltk
from nltk.corpus import stopwords
import string
import json
from langchain_openai import OpenAIEmbeddings
import tiktoken
import requests
import re
import time
import uuid
from pinecone import Pinecone, PodSpec

TAG_RE = re.compile(r"<[^>]+>")


def remove_stopwords(text):
    # Tokenize the text into words
    words = nltk.word_tokenize(text)
    punc = string.punctuation + "’“”."

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    filtered_words = [word.lower() for word in words if word not in stop_words]
    filtered_words = [word for word in filtered_words if word not in punc]

    # Join the filtered words back into a single string
    processed_text = " ".join(filtered_words)

    return processed_text


def strip_html_tags(text):
    return TAG_RE.sub(" ", text)


def scrape_confluence_space(
    confluence_hostname, username, api_key, space_key, start_at=0, limit=100
):
    page_data = []
    encoding = tiktoken.get_encoding("cl100k_base")
    max_tokens = 8191

    response = requests.get(
        "https://" + confluence_hostname + "/wiki/rest/api/content",
        params={
            "limit": 100,
            "start": start_at,
            "spaceKey": space_key,
            "expand": "body.storage",
        },
        auth=(username, api_key),
    )

    if response.status_code == 200:
        data = response.json()
        for page in data["results"]:
            body = strip_html_tags(page["body"]["storage"]["value"])
            tokens = encoding.encode(body)[:max_tokens]
            body = encoding.decode(tokens)
            page_dict = {
                "title": page["title"],
                "text": body,
                "url": confluence_hostname + "/wiki" + page["_links"]["webui"],
            }
            page_data.append(page_dict)

        if data["size"] == data["limit"]:
            page_data += scrape_confluence_space(
                confluence_hostname,
                username,
                api_key,
                space_key,
                start_at + limit,
                limit,
            )

        return page_data
    else:
        print(f"Failed to get data from Confluence: {response.status_code}")
        print(response.text)
        return None


with open("credentials.json") as credentials_file:
    credentials = json.loads(credentials_file.read())

spacename = "D61"

direct_docs = scrape_confluence_space(
    credentials["CONFLUENCE_HOSTNAME"],
    credentials["CONFLUENCE_USERNAME"],
    credentials["CONFLUENCE_API_KEY"],
    spacename,
)

print(f"Scraped {len(direct_docs)} documents from Confluence")

embeddings_model = OpenAIEmbeddings(openai_api_key=credentials["OPENAI_API_KEY"])
embeddings = embeddings_model.embed_documents([i["text"] for i in direct_docs])

print(f"Embedded {len(embeddings)} documents")

dataset = list(
    map(
        lambda i: {
            "id": str(uuid.uuid4()),
            "metadata": i[0],
            "values": i[1],
        },
        zip(direct_docs, embeddings),
    )
)

pc = Pinecone(api_key=credentials["PINECONE_API_KEY"])
spec = PodSpec(environment="gcp-starter", pod_type="starter", pods=1)

# check for and delete index if already exists
index_name = "support-agent-" + spacename.lower()
if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)

print(f"Creating new index {index_name}")
# we create a new index
pc.create_index(
    index_name,
    dimension=1536,  # dimensionality of text-embedding-ada-002
    metric="dotproduct",
    spec=spec,
)

while not pc.describe_index(index_name).status["ready"]:
    time.sleep(1)

index = pc.Index(index_name)
print(index.describe_index_stats())

batch_size = 100
for i in range(0, len(dataset), batch_size):
    index.upsert(dataset[i : i + batch_size])
