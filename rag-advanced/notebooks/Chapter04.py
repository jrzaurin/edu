import asyncio

import cohere
import weave
from IPython.display import Markdown
from scripts.query_enhancer import QueryEnhancer
from scripts.rag_pipeline import QueryEnhancedRAGPipeline, SimpleRAGPipeline
from scripts.response_generator import (
    QueryEnhanedResponseGenerator,
    SimpleResponseGenerator,
)
from scripts.response_metrics import LLM_METRICS
from scripts.retriever import BM25Retriever

# from scripts.tokens_and_api_keys import COHERE_API_KEY


WANDB_PROJECT = "rag-course"

weave_client = weave.init(WANDB_PROJECT)

# Reload the data from Chapter 3
chunked_data = weave.ref("chunked_data:v0").get()

# print(chunked_data.rows[:2])

query_enhancer = QueryEnhancer()

response = asyncio.run(
    query_enhancer.predict("How do I log images in lightning with wandb?")
)


retriever = BM25Retriever()
retriever.index_data(chunked_data.rows)


# lets add the new prompt
QUERY_ENHANCED_PROMPT = open("prompts/query_enhanced_system.txt").read()

response_generator = QueryEnhanedResponseGenerator(
    model="command-r", prompt=QUERY_ENHANCED_PROMPT
)

query_enhanced_rag_pipeline = QueryEnhancedRAGPipeline(
    query_enhancer=query_enhancer,
    retriever=retriever,
    response_generator=response_generator,
    top_k=2,
)

response = asyncio.run(
    query_enhanced_rag_pipeline.predict("How do I log images in lightning with wandb?")
)


Markdown(response)

eval_dataset = weave.ref(
    "weave:///rag-course/dev/object/Dataset:Qj4IFICc2EbdXu5A5UuhkPiWgxM1GvJMIvXEyv1DYnM"
).get()

print(eval_dataset.rows[:2])

# Let also initialize the baseline RAG pipeline from chapter 3


INITIAL_PROMPT = open("prompts/initial_system.txt", "r").read()
response_generator = SimpleResponseGenerator(model="command-r", prompt=INITIAL_PROMPT)
simple_rag_pipeline = SimpleRAGPipeline(
    retriever=retriever, response_generator=response_generator, top_k=5
)


response_evaluations = weave.Evaluation(
    name="Response_Evaluation",
    dataset=eval_dataset,
    scorers=LLM_METRICS,
    preprocess_model_input=lambda x: {"query": x["question"]},
)

baseline_response_scores = asyncio.run(
    response_evaluations.evaluate(simple_rag_pipeline)
)

query_enhanced_response_scores = asyncio.run(
    response_evaluations.evaluate(query_enhanced_rag_pipeline)
)
