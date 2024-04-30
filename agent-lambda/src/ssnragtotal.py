from typing import Any
import dspy
import json
import openai
from dspy.retrieve.pinecone_rm import PineconeRM
from runpod_lm import OllamaRunpod


class GenerateAnswer(dspy.Signature):
    """You are a customer support agent helping customers with problems they encounter using a program called DIRECT.
    This is not related to DirectX so do not talk about DirectX
    The request messages will fall into one of four categories and you should respond accordingly:
    one. A request for account setup or password reset and other account-related issues. In this case state that the support team has been notified and they will hear back soon.
    two. Messages from support agents responding to a customer's request. In this case, you should respond "Agent Message"
    three. Messages from customers thanking the support team. In this case, you should respond "Customer Thank You"
    four. A question about the software or a problem encountered with the software, in this case,
    you should respond with the answer to the question.
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Be descriptive and concise, while also presenting all options available to the customer.
    """

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField(desc="The support request or question from a customer")
    answer = dspy.OutputField()


class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(answer=prediction.answer)


class AnswerTicket:
    def __init__(self, openai_key: str, runpod_key: str, pinecone_key: str) -> None:

        openai.api_key = openai_key

        llm = OllamaRunpod(
            model="custom",
            timeout_s=600,
            max_tokens=512,
            num_ctx=2048,
            top_k=10,
            top_p=0.95,
            base_url="https://api.runpod.ai/v2/cgoxys1krx1ref/runsync",
            runpod_token=runpod_key,
            temperature=0.6,
        )

        retriever_model = PineconeRM(
            "support-agent-d61",
            pinecone_key,
            "gcp-starter",
            openai_api_key=openai_key,
            k=1,
        )
        dspy.settings.configure(lm=llm, rm=retriever_model)
        self.ssnrag = RAG()
        # third_compiled_rag.load("jlines_compiled_rag_test")

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        prediction = self.ssnrag(args[0])
        return prediction.answer
