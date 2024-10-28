import operator
from typing import Annotated, Optional

import dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.documents import Document
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from llqm.modules.llm_wrapper import chat_completion
from llqm.schemas.schemas import HistoryResponse, SearchQueries

dotenv.load_dotenv()
TOP_N_SEARCH_QUERIES = 5
TOP_N_SEARCH_RESULTS = 5


class GraphState(BaseModel):
    url: str
    original_doc: Optional[Document] = None
    history_queries: Optional[SearchQueries] = None
    history_urls: Optional[list[str]] = None
    history_info: Annotated[Optional[list[HistoryResponse]], operator.add] = None
    history_summary: Optional[str] = None


def node_init(state: GraphState):
    url = state.url
    loader = WebBaseLoader(url)
    docs = loader.load()
    return {"original_doc": docs[0]}


def node_history_queries(state: GraphState):
    doc = state.original_doc
    sys_prompt = f"""You will be given a document. Your task is to generate {TOP_N_SEARCH_QUERIES} search \
            queries that can help find information about events preceding the document. Specifically, \
            the queries should aim to:
            1. Explain what happened before the events described in the document.
            2. Identify the precedents of the events in the document.
            3. Determine the causes of the events in the document.
            4. Historical context of the events in the document."""
    user_prompt = f"document: {doc}"
    history_queries = chat_completion(
        sys_prompt, user_prompt, response_format=SearchQueries
    )
    return {"history_queries": history_queries}


def node_history_urls(state: GraphState):
    queries = state.history_queries.queries
    search = GoogleSerperAPIWrapper(type="news")
    urls = set()
    for query in queries:
        urls = urls | set(
            [
                result["link"]
                for result in search.results(query)["news"][:TOP_N_SEARCH_RESULTS]
            ]
        )
    return {"history_urls": list(urls)}


def distribute_urls(state: GraphState):
    urls = state.history_urls
    return [Send("node_history_extract", url) for url in urls]


def node_history_extract(url):
    loader = WebBaseLoader(url, requests_kwargs={"timeout": 15}, raise_for_status=True)
    try:
        docs = loader.load()
        system_prompt = """You will be given a document, and your task is to extract the following information:
        1. The main event happend time, be aware this is not the publish time, it's the event's time only.
        2. The article's publish time, this is the time the article was published, not the event's time.
        3. The summary of the document."""
        user_prompt = f"document: {docs[0]}"
        response = [
            chat_completion(system_prompt, user_prompt, response_format=HistoryResponse)
        ]
    except Exception as e:
        print(e)
        response = []
    return {"history_info": response}


def node_history_summary(state: GraphState):
    history_info = state.history_info
    system_prompt = """You will be given an original document along with a list of historical documents\
        related to it. Your task is to write a summary that explains the historical context to help\
        understand the event described in the original document. You may use up to 5 most related and \
        importnat documents to write the summary. At the end of the summary, include the URLs of the \
        sources you used, which are provided in the documents."""
    user_prompt = (
        f"original document: {state.original_doc}, history documents: {history_info}"
    )
    history_summary = chat_completion(system_prompt, user_prompt)
    print(history_summary)
    return {"history_summary": history_summary}


graph = StateGraph(GraphState)
graph.add_node(node_init.__name__, node_init)
graph.add_node(node_history_queries.__name__, node_history_queries)
graph.add_node(node_history_urls.__name__, node_history_urls)
graph.add_node(node_history_extract.__name__, node_history_extract)
graph.add_node(node_history_summary.__name__, node_history_summary)
graph.add_edge(START, node_init.__name__)
graph.add_edge(node_init.__name__, node_history_queries.__name__)
graph.add_edge(node_history_queries.__name__, node_history_urls.__name__)
graph.add_conditional_edges(
    node_history_urls.__name__, distribute_urls, [node_history_extract.__name__]
)
graph.add_edge(node_history_extract.__name__, node_history_summary.__name__)
graph.add_edge(node_history_summary.__name__, END)
app = graph.compile()

url = "https://msutoday.msu.edu/news/2024/msu-new-plant-and-environmental-sciences-building-crtiical-to-advancing-climate-resilient-plants"
app.invoke({"url": url})
