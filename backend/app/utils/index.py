import logging
import os

from llama_index import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    ServiceContext,
    OpenAIEmbedding
)
from llama_index.llms import OpenRouter

key = os.getenv("LLM_KEY")
oa_key = os.getenv("OPENAI_KEY")

embedding = OpenAIEmbedding(api_key=oa_key)


STORAGE_DIR = "./storage"  # directory to cache the generated index
DATA_DIR = "./data"  # distreaming_response.print_response_stream()rectory containing the documents to index

OpenAIEmbedding
service_context = ServiceContext.from_defaults(
    llm=OpenRouter(
        api_key=key,
        max_tokens=2000,
        context_window=4096,
        model="mistralai/mixtral-8x7b-instruct",
    ),
    embed_model=embedding
)


def get_index():
    logger = logging.getLogger("uvicorn")
    # check if storage already exists
    if not os.path.exists(STORAGE_DIR):
        logger.info("Creating new index")
        # load the documents and create the index
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        index = VectorStoreIndex.from_documents(
            documents, service_context=service_context
        )
        # store it for later
        index.storage_context.persist(STORAGE_DIR)
        logger.info(f"Finished creating new index. Stored in {STORAGE_DIR}")
    else:
        # load the existing index
        logger.info(f"Loading index from {STORAGE_DIR}...")
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        index = load_index_from_storage(
            storage_context, service_context=service_context
        )
        logger.info(f"Finished loading index from {STORAGE_DIR}")
    return index
