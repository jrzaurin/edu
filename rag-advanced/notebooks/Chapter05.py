import asyncio
# import getpass
# import os

import nest_asyncio
import weave

# from scripts.reranker import CohereReranker
from scripts.retrieval_metrics import IR_METRICS
from scripts.retriever import (
    DenseRetriever,
    DenseRetrieverWithReranker,
    HybridRetrieverReranker,
)
# from scripts.utils import display_source


# os.environ["COHERE_API_KEY"] = getpass.getpass("Please enter your COHERE_API_KEY")


nest_asyncio.apply()


WANDB_PROJECT = "rag-course"

weave_client = weave.init(WANDB_PROJECT)


# We will download the chunked data from chapter 3


chunked_data = weave.ref("chunked_data:v0").get()

# chunked_data.rows[:2]


# ## Embedding based retriever

# Let's initialize the `DenseRetriever` and index the data.


dense_retriever = DenseRetriever()
dense_retriever.index_data(chunked_data.rows)


eval_dataset = weave.ref(
    "weave:///rag-course/dev/object/Dataset:Qj4IFICc2EbdXu5A5UuhkPiWgxM1GvJMIvXEyv1DYnM"
).get()

retrieval_evaluation = weave.Evaluation(
    name="Dense Retrieval Evaluation",
    dataset=eval_dataset,
    scorers=IR_METRICS,
    preprocess_model_input=lambda x: {"query": x["question"], "k": 10},
)

dense_retrieval_scores = asyncio.run(retrieval_evaluation.evaluate(dense_retriever))


# Let's initialize the `DenseRetrieverWithReranker` and index the data.


dense_retriever_rerank = DenseRetrieverWithReranker()
dense_retriever_rerank.index_data(chunked_data.rows)


eval_dataset = weave.ref(
    "weave:///rag-course/dev/object/Dataset:Qj4IFICc2EbdXu5A5UuhkPiWgxM1GvJMIvXEyv1DYnM"
).get()

retrieval_evaluation = weave.Evaluation(
    name="Dense Retrieval Rerank Evaluation",
    dataset=eval_dataset,
    scorers=IR_METRICS,
    preprocess_model_input=lambda x: {"query": x["question"], "top_k": 20, "top_n": 10},
)

dense_retrieval_scores = asyncio.run(
    retrieval_evaluation.evaluate(dense_retriever_rerank)
)


# ## Hybrid Retriever

hybrid_retriever = HybridRetrieverReranker()

hybrid_retriever.index_data(chunked_data.rows)


eval_dataset = weave.ref(
    "weave:///rag-course/dev/object/Dataset:Qj4IFICc2EbdXu5A5UuhkPiWgxM1GvJMIvXEyv1DYnM"
).get()

retrieval_evaluation = weave.Evaluation(
    name="Dense Retrieval Rerank Evaluation",
    dataset=eval_dataset,
    scorers=IR_METRICS,
    preprocess_model_input=lambda x: {"query": x["question"], "top_k": 20, "top_n": 10},
)

hybrid_retrieval_scores = asyncio.run(retrieval_evaluation.evaluate(hybrid_retriever))
