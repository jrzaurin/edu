import asyncio

import nest_asyncio
import weave
from scripts.rag_pipeline import SimpleRAGPipeline
from scripts.response_generator import SimpleResponseGenerator
from scripts.response_metrics import LLM_METRICS, NLP_METRICS
from scripts.retrieval_metrics import IR_METRICS
from scripts.retriever import TFIDFRetriever

nest_asyncio.apply()


WANDB_PROJECT = "rag-course"

weave_client = weave.init(WANDB_PROJECT)


# Easy eval dataset with 20 samples.
eval_dataset = weave.ref(
    "weave:///rag-course/dev/object/Dataset:Qj4IFICc2EbdXu5A5UuhkPiWgxM1GvJMIvXEyv1DYnM"
).get()
print("Number of evaluation samples: ", len(eval_dataset.rows))


# Reload the data from Chapter 1
chunked_data = weave.ref("chunked_data:v0").get()
print("Number of chunked data: ", len(chunked_data.rows))

retriever = TFIDFRetriever()
retriever.index_data(chunked_data.rows)

# #### Evaluating retrieval on other metrics
retrieval_evaluation = weave.Evaluation(
    name="Retrieval_Evaluation",
    dataset=eval_dataset.rows,
    scorers=IR_METRICS,
    preprocess_model_input=lambda x: {"query": x["question"], "k": 5},
)
retrieval_scores = asyncio.run(retrieval_evaluation.evaluate(retriever))


# ### Using an LLM evaluator for evaluating retriever
#
# **ref: https://arxiv.org/pdf/2406.06519**
retrieval_evaluation = weave.Evaluation(
    name="LLM_Judge_Retrieval_Evaluation",
    dataset=eval_dataset,
    scorers=LLM_METRICS,
    preprocess_model_input=lambda x: {"query": x["question"], "k": 5},
)
retrieval_scores = asyncio.run(retrieval_evaluation.evaluate(retriever))


# ## Evaluating the Response
INITIAL_PROMPT = open("prompts/initial_system.txt", "r").read()
response_generator = SimpleResponseGenerator(
    model="command-r-plus", prompt=INITIAL_PROMPT
)
rag_pipeline = SimpleRAGPipeline(
    retriever=retriever, response_generator=response_generator, top_k=5
)

response_evaluations = weave.Evaluation(
    name="Response_Evaluation",
    dataset=eval_dataset,
    scorers=NLP_METRICS[:-1],
    preprocess_model_input=lambda x: {"query": x["question"]},
)
response_scores = asyncio.run(response_evaluations.evaluate(rag_pipeline))


# ### Using an LLM as a Response Judge
#
# Some metrics cannot be defined objectively and are particularly useful for more subjective or complex criteria.
# We care about correctness, faithfulness, and relevance.
#
# - **Answer Correctness** - Is the generated answer correct compared to the reference and thoroughly answers the user's query?
# - **Answer Relevancy** - Is the generated answer relevant and comprehensive?
# - **Answer Factfulness** - Is the generated answer factually consistent with the context document?
#
correctness_evaluations = weave.Evaluation(
    name="Correctness_Evaluation",
    dataset=eval_dataset,
    scorers=LLM_METRICS,
    preprocess_model_input=lambda x: {"query": x["question"]},
)
response_scores = asyncio.run(correctness_evaluations.evaluate(rag_pipeline))


# ## Exercise
#
# 1. Implement the `Relevance` and `Faithfulness` evaluators and evaluate the pipeline on all the dimensions.
# 2. Generate and share a W&B report with the following sections in the form of tables and charts:
#
#     - Summary of the evaluation
#     - Retreival Evaluations
#         - IR Metrics
#         - LLM As a Retrieval Judge Metric
#     - Response Evalations
#         - Traditional NLP Metrics
#         - LLM Judgement Metrics
#     - Overall Evalations
#     - Conclusion
#
