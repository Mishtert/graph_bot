import os

from langchain.vectorstores.faiss import FAISS
from multi_skill_bot.utils.load_model import embeddings

from multi_skill_bot.utils.litm_utils import read_create_docs

# Get the current working directory
# base_dir = os.getcwd()

# Construct the path dynamically
# config_path = os.path.join(base_dir, "multi_skill_bot", "config.env")
enterprise_documents = read_create_docs()



def embed_index(doc_list, embed_fn, index_store):
    """Function takes in existing vector_store,
    new doc_list and embedding function that is
    initialized on appropriate model. Local or online.
    New embedding is merged with the existing index. If no
    index given a new one is created"""
    # check whether the doc_list is documents, or text
    try:
        faiss_db = FAISS.from_documents(doc_list,
                                        embed_fn)
    except Exception as e:
        faiss_db = FAISS.from_texts(doc_list,
                                    embed_fn)

    if os.path.exists(index_store):
        local_db = FAISS.load_local(index_store, embed_fn)
        # merging the new embedding with the existing index store
        local_db.merge_from(faiss_db)
        print("Merge completed")
        local_db.save_local(index_store)
        print("Updated index saved")
    else:
        faiss_db.save_local(folder_path=index_store)
        print("New store created...")


def get_docs_length(index_path, embed_fn):
    test_index = FAISS.load_local(index_path,
                                  embeddings=embed_fn)
    test_dict = test_index.docstore._dict
    return len(test_dict.values())


embed_index(doc_list=enterprise_documents,
            embed_fn=embeddings,
            index_store='./multi_skill_bot/data/erc_docs')

get_docs_length(index_path='./multi_skill_bot/data/erc_docs', embed_fn=embeddings)

test_index = FAISS.load_local("./multi_skill_bot/data/erc_docs", embeddings)

test_index.similarity_search("dental plan")
