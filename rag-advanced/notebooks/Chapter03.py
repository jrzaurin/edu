import asyncio

# import nltka
import numpy as np
import weave
from scripts.chunking import chunk_documents
from scripts.preprocess import convert_contents_to_text, make_text_tokenization_safe
from scripts.rag_pipeline import SimpleRAGPipeline
from scripts.response_generator import SimpleResponseGenerator
from scripts.response_metrics import ALL_METRICS as RESPONSE_METRICS
from scripts.retrieval_metrics import ALL_METRICS as RETRIEVAL_METRICS
from scripts.retriever import BM25Retriever, TFIDFRetriever
from scripts.utils import (
    TOKENIZERS,
    display_source,
    get_special_tokens_set,
    length_function,
    tokenize_text,
)

# import getpass
# import os

# from .tokens_and_api_keys import COHERE_API_KEY

# nltk.download("wordnet")


# os.environ["COHERE_API_KEY"] = getpass.getpass("Please enter your COHERE_API_KEY")


WANDB_PROJECT = "rag-course"

weave_client = weave.init(WANDB_PROJECT)

# download the `raw_data` Dataset
raw_data = weave.ref("raw_data:v0").get()

# this is how we index into the data
# print(raw_data.rows[:2])
raw_data = list(map(dict, raw_data.rows[:]))

# # this is the function that will tokenize the text
# display_source(tokenize_text)
# # this is the function that calculates the number of tokens
# display_source(length_function)

for doc in raw_data[:]:
    doc["metadata"]["words"] = doc["metadata"].pop("raw_tokens")
    doc["metadata"]["tokens"] = length_function(doc["content"])
# raw_data[:2]

# # this is the function that converts the markdown to html
# display_source(convert_contents_to_text)
# # this is the function that cleans the text
# display_source(make_text_tokenization_safe)

special_tokens_set = get_special_tokens_set(TOKENIZERS["command-r"])
print(list(special_tokens_set)[:5])

parsed_data = []
for doc in raw_data:
    parsed_doc = doc.copy()
    content = convert_contents_to_text(doc["content"])
    parsed_doc["parsed_content"] = make_text_tokenization_safe(
        content, special_tokens_set=special_tokens_set
    )
    parsed_doc["metadata"]["parsed_tokens"] = length_function(
        parsed_doc["parsed_content"]
    )
    parsed_data.append(parsed_doc)
# parsed_data[:2]

# build weave dataset
parsed_data = weave.Dataset(name="parsed_data", rows=parsed_data)

# publish the dataset
weave.publish(parsed_data)

# download the `parsed_data` Dataset
parsed_data = weave.ref("parsed_data:v0").get()

# # this is how we index into the data
# print(parsed_data.rows[:2])

parsed_data = list(map(dict, parsed_data.rows[:]))
# parsed_data[:2]

# # this is the function that chunks the documents
# display_source(chunk_documents)

chunked_data = chunk_documents(parsed_data)
# chunked_data[:2]

mean_chunk_size = np.mean([doc["metadata"]["parsed_tokens"] for doc in chunked_data])
std_chunk_size = np.std([doc["metadata"]["parsed_tokens"] for doc in chunked_data])
print(f"Mean chunk size: {mean_chunk_size}, Std chunk size: {std_chunk_size}")

# Again, we'll store the chunked data in a weave Dataset
chunked_data = weave.Dataset(name="chunked_data", rows=chunked_data)

# publish the dataset
weave.publish(chunked_data)


display_source(BM25Retriever)  # fetch the chunked data
chunked_data = weave.ref("chunked_data:v0").get()

# this is how we index into the data
print(chunked_data.rows[:2])

chunked_data = list(map(dict, chunked_data.rows[:]))

bm25_retriever = BM25Retriever()
bm25_retriever.index_data(chunked_data)

tfidf_retriever = TFIDFRetriever()
tfidf_retriever.index_data(chunked_data)

INITIAL_PROMPT = open("prompts/initial_system.txt", "r").read()
response_generator = SimpleResponseGenerator(model="command-r", prompt=INITIAL_PROMPT)
bm25_rag_pipeline = SimpleRAGPipeline(
    retriever=bm25_retriever, response_generator=response_generator, top_k=5
)
tfidf_rag_pipeline = SimpleRAGPipeline(
    retriever=tfidf_retriever, response_generator=response_generator, top_k=5
)


eval_dataset = weave.ref(
    "weave:///rag-course/dev/object/Dataset:Qj4IFICc2EbdXu5A5UuhkPiWgxM1GvJMIvXEyv1DYnM"
).get()

print("Number of evaluation samples: ", len(eval_dataset.rows))

retrieval_evaluation = weave.Evaluation(
    name="Retrieval_Evaluation",
    dataset=eval_dataset.rows,
    scorers=RETRIEVAL_METRICS,
    preprocess_model_input=lambda x: {"query": x["question"], "k": 5},
)
bm25_retrieval_scores = asyncio.run(retrieval_evaluation.evaluate(bm25_retriever))
tfidf_retrieval_scores = asyncio.run(retrieval_evaluation.evaluate(tfidf_retriever))

response_evaluations = weave.Evaluation(
    name="Response_Evaluation",
    dataset=eval_dataset.rows,
    scorers=RESPONSE_METRICS,
    preprocess_model_input=lambda x: {"query": x["question"]},
)
bm25_response_scores = asyncio.run(response_evaluations.evaluate(bm25_rag_pipeline))
tfidf_response_scores = asyncio.run(response_evaluations.evaluate(tfidf_rag_pipeline))
