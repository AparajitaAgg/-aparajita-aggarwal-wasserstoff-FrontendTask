from flask import Flask, request
from bs4 import BeautifulSoup
import requests
from datasets import Dataset, concatenate_datasets
import pandas as pd
from transformers import BertTokenizer, BertModel, RagRetriever, RagTokenizer, pipeline, AutoTokenizer

model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = BertTokenizer.from_pretrained(model_ckpt)
model = BertModel.from_pretrained(model_ckpt)
tokenizer2 = AutoTokenizer.from_pretrained("distilbert-base-uncased")
embeddings_dataset=None
query_dataset=None

app = Flask(__name__)

@app.route("/embed-new-post", methods=["POST"])
def embed_new_post():
    new_url=request.args["new_url"]
    post_id= request.args["post_id"]
    response = requests.get(new_url)
    create_post_embedding(response)

@app.route("/post-user-input", methods=["POST"])
def receive_user_input():
    user_input=request.args['user_input']
    previous_context=retrieve_augment_chat_context(user_input)
    initial_response= rag_generate_response(user_input)
    response=generate_response(previous_context, initial_response)
    return response

#utility methods
def create_post_embedding(response):
    soup = BeautifulSoup(response.content, "html.parser")
    text = soup.get_text()
    sentences=text.split(".")
    sentences=[sentence.strip() for sentence in sentences ]
    sentence_df=pd.DataFrame(sentences, columns=["sentence"])
    sentences_dataset = Dataset.from_pandas(sentence_df)
    embeddings_dataset = sentences_dataset.map(lambda x: {"embeddings": get_embeddings(x["sentence"]).detach().cpu().numpy()[0]})
    embeddings_dataset.add_faiss_index(column="embeddings")

def generate_response(user_input, context):
    inputs = tokenizer2(
        context,
        user_input,
        max_length=384,
        padding="max_length",
    )
    return inputs

def rag_generate_response(user_input):
    retriever = RagRetriever.from_pretrained("facebook/rag-token-base", retriever="bm25")
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")
    generator = pipeline('text-generation', model='facebook/rag-token-base')
    question_embedding = get_embeddings([user_input]).cpu().detach().numpy()
    scores, samples = embeddings_dataset.get_nearest_examples(
    "embeddings", question_embedding, k=5)
    samples_df = pd.DataFrame.from_dict(samples)
    samples_df["scores"] = scores
    context=samples_df["sentence"].str.cat(sep=". ")
    input_dict = tokenizer(context, user_input, return_tensors="pt")
    retrieved_docs = retriever(input_dict["input_ids"])
    generated_text = generator(retrieved_docs["context_input_ids"])[0]["generated_text"]
    return generated_text

def retrieve_augment_chat_context(user_input):
    question_embedding = get_embeddings([user_input]).cpu().detach().numpy()
    scores, samples = query_dataset.get_nearest_examples(
    "embeddings", question_embedding, k=3)
    samples_df = pd.DataFrame.from_dict(samples)
    samples_df["scores"] = scores
    query_string=samples_df["query"].str.cat(sep=". ")
    return query_string

def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

def get_embeddings(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    model_output = model(**encoded_input)
    return cls_pooling(model_output)

