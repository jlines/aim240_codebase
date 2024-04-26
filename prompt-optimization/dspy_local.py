# Connect to Weaviate Retriever and configure LLM
import dspy
from dspy.retrieve.pinecone_rm import PineconeRM
from dspy.teleprompt import BayesianSignatureOptimizer, MIPRO, BootstrapFewShot, COPRO
from dspy.evaluate.evaluate import Evaluate
from ssnrag import RAG

# from pinecone import Pinecone
import openai
import json

credentials = {}
with open("credentials.json") as credentials_file:
    credentials = json.loads(credentials_file.read())

openai.api_key = credentials["OPENAI_API_KEY"]

llm = dspy.OllamaLocal(
    model="ssn",
    timeout_s=600,
    max_tokens=512,
    num_ctx=2048,
    top_k=10,
    top_p=0.95,
    temperature=0.6,
)

# ollamaLLM = dspy.OpenAI(api_base="http://localhost:11434/v1/", api_key="ollama", model="mistral-7b-instruct-v0.2-q6_K", stop='\n\n', model_type='chat')
# Thanks Knox! https://twitter.com/JR_Knox1977/status/1756018720818794916/photo/1

# pc = Pinecone(api_key=credentials["PINECONE_API_KEY"])
retriever_model = PineconeRM(
    "support-agent-d61",
    credentials["PINECONE_API_KEY"],
    "gcp-starter",
    openai_api_key=credentials["OPENAI_API_KEY"],
)
# Assumes the Weaviate collection has a text key `content`
dspy.settings.configure(lm=llm, rm=retriever_model)

question = "How can I reduce the size of my patches?"
"""
print(dspy.settings.lm(question))
context_example = dspy.OpenAI(model="gpt-4")

with dspy.context(llm=context_example):
    print(context_example(question))
"""


def load_tickets():
    import pandas as pd

    # Specify the file path
    file_path = "data/CleanedJira.csv"

    # Load the CSV file into a pandas dataframe
    df = pd.read_csv(file_path)

    # Print the summary column sorted and unique
    # tickets = df["Issue Key"].unique()

    # get the first record by issue key ordered by date
    tickets = df.sort_values("Date").groupby("Issue key").first().reset_index()

    # remove the records from tickets where SupportResponse is True
    tickets = tickets[tickets["SupportResponse"] != True]

    # remove the records from tickets where Summary contains "hello"
    tickets = tickets[
        ~tickets["Summary"].str.contains("DIRECT 6 Evaluation", case=True)
    ]

    out = (tickets["Summary"] + " " + tickets["Message"]).tolist()

    return out


# Parsing the markdown content to get only questions
tickets = load_tickets()

# Displaying the first few extracted questions
# print(tickets[:5])  # Displaying only the first few for brevity
print(len(tickets))

# ToDo, add random splitting -- maybe wrap this entire thing in a cross-validation loop
trainset = tickets[32:33]  # examples for training
devset = tickets[100:110]  # examples for development
testset = tickets[150:160]  # examples for testingexity

trainset = [
    dspy.Example(question=question).with_inputs("question") for question in trainset
]
devset = [
    dspy.Example(question=question).with_inputs("question") for question in devset
]
testset = [
    dspy.Example(question=question).with_inputs("question") for question in testset
]

metricLM = dspy.OpenAI(model="gpt-3.5-turbo", max_tokens=1000, model_type="chat")

# Signature for LLM assessments.


class Assess(dspy.Signature):
    """Assess the quality of an answer to a question."""

    context = dspy.InputField(desc="The context for answering the question.")
    assessed_question = dspy.InputField(desc="The evaluation criterion.")
    assessed_answer = dspy.InputField(desc="The answer to the question.")
    assessment_answer = dspy.OutputField(
        desc="A rating between 1 and 5. Only output the rating and nothing else."
    )


def llm_metric(gold, pred, trace=None):
    predicted_answer = pred.answer
    question = gold.question

    print(f"Test Question: {question}")
    print(f"Predicted Answer: {predicted_answer}")

    detail = "Is the assessed answer detailed?"
    faithful = "Is the assessed text grounded in the context? Say no if it includes significant facts not in the context."
    overall = f"Please rate how well this answer answers the question, `{question}` based on the context.\n `{predicted_answer}`"

    with dspy.context(lm=metricLM):
        # context = dspy.Retrieve()(question).passages
        r = dspy.Retrieve()
        res = r(question)
        context = res.passages

        detail = dspy.ChainOfThought(Assess)(
            context="N/A", assessed_question=detail, assessed_answer=predicted_answer
        )
        faithful = dspy.ChainOfThought(Assess)(
            context=context,
            assessed_question=faithful,
            assessed_answer=predicted_answer,
        )
        overall = dspy.ChainOfThought(Assess)(
            context=context, assessed_question=overall, assessed_answer=predicted_answer
        )

    print(f"Faithful: {faithful.assessment_answer}")
    print(f"Detail: {detail.assessment_answer}")
    print(f"Overall: {overall.assessment_answer}")

    total = (
        float(detail.assessment_answer)
        + float(faithful.assessment_answer) * 2
        + float(overall.assessment_answer)
    )

    return total / 5.0


"""
EXPLAIN HOW THE ANSWER VALIDATION STUFF WORKS

test_example = dspy.Example(question="How can I reduce the size of my patches?")
test_pred = dspy.Example(answer="By using the delta update paths feature of DIRECT")

type(llm_metric(test_example, test_pred))

metricLM.inspect_history(n=3)
"""


llm_prompter = dspy.OllamaLocal(
    model="ssn", timeout_s=600, max_tokens=2000, model_type="chat"
)

llm_prompter = dspy.OpenAI(model="gpt-3.5-turbo", max_tokens=1000, model_type="chat")

opt_MIPRO = MIPRO(
    task_model=dspy.settings.lm,
    metric=llm_metric,
    prompt_model=llm_prompter,
    num_candidates=5,
    verbose=False,
)

opt_COPRO = COPRO(
    task_model=dspy.settings.lm,
    metric=llm_metric,
    prompt_model=llm_prompter,
    num_candidates=5,
    verbose=False,
)

opt_bootstrap = BootstrapFewShot(
    metric=llm_metric,
)


kwargs = dict(num_threads=1, display_progress=False, display_table=0)
third_compiled_rag = opt_COPRO.compile(
    RAG(),
    trainset=trainset,
    eval_kwargs=kwargs,
)

""" For MIPRO
kwargs = dict(num_threads=1, display_progress=False, display_table=0)
third_compiled_rag = opt_MIPRO.compile(
    RAG(),
    trainset=trainset,
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
    eval_kwargs=kwargs,
)
"""

""" For BootstrapFewShot

third_compiled_rag = opt_bootstrap.compile(
    RAG(),
    trainset=trainset,
)"""

# print("PROMPERT", llm_prompter.inspect_history(n=1))
# quit()

print("!!!!!!!!!!!!!!!!!!! FINISHED TRAINING !!!!!!!!!!!!!!!!!!!")
third_compiled_rag.save("jlines_compiled_rag_test")

# Set up the `evaluate_on_hotpotqa` function. We'll use this many times below.
evaluate = Evaluate(
    devset=testset, num_threads=1, display_progress=True, display_table=5
)
p = evaluate(third_compiled_rag, metric=llm_metric)
