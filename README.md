# AIM240 Customer Support Agent - RAG System

This project contains several components which are organized in different folders. Here's a brief description of each:

## datapipeline

This folder contains scripts and files related to the data pipeline of the project.

[doc_preproc_vectorize_load.py](datapipeline/doc_preproc_vectorize_load.py)
1. Scrapes documents from a Confluence space.
2. Preprocesses the scraped documents by removing HTML tags and stopwords.
3. Embeds the preprocessed documents using an OpenAI embeddings model.
4. Creates a new index in Pinecone and upserts the embedded documents into the index.


## finetuning

This folder contains scripts and files for fine-tuning a machine learning model.

[train-ec2.py](finetuning/train-ec2.py)
Loads the model 'NousResearch/Llama-2-7b-chat-hf' from hugging face and finetunes it on a dataset that is a mixture of the customer support data set called 'CheshireAI/guanaco-unchained' and data scraped from the company Jira support tickets.

[convert-ggml.sh](finetuning/convert-ggml.sh)
Converts the model and fine tuned LoRa adapter to a GGML file for use with a LLM hosting program such as Llama.cpp or Ollama.


## prompt-optimization

This folder is dedicated to the optimization of prompts for a language model. It contains scripts for generating, evaluating, and optimizing prompts.

## agent-lambda

This folder contains the implementation of a RAG (Retrieval-Augmented Generation) model with a lambda architecture. This includes scripts for the retrieval system, and generation system.

[app.py](agent-lambda/src/app.py)
Reads the question from a ticket in Jira, passes that question into the RAG system, then replies to the ticket with the generated answer.

[runpod_lm.py](agent-lambda/src/runpod_lm.py)
Custom LLM module for DSPy which can interface with the custom API used to host the Ollama instance on the Runpod serverless platform.

[ssnragtotal.py](agent-lambda/src/ssnragtotal.py)
The definition class for the RAG system. This defines the components used to collect information and the techniques used to create embeddings to query the vector db as well as generating the optimal propmt. This file has a lot of potential for improvement of the system.

[Dockerfile](agent-lambda/src/Dockerfile)
Since the DSPy library is larger than the 250mb limit of standard lambda functions, I must deploy as a docker image which can be larger

## runpod-docker

This folder contains Dockerfile(s) and scripts to build and run a Docker container which allows the fine tuned LLM Model to run on the RunPod Serverless platform.

[main.py](runpod-docker/src/main.py)
This connects the Ollama local API with the hosted API running on Runpod and passes information between the two.
