from pydantic import BaseModel, Field


class SearchQueries(BaseModel):
    queries: list[str]


class HistoryResponse(BaseModel):
    event_time: str = Field(description="The time of the event happened")
    news_time: str = Field(description="The time of the news published")
    title: str = Field(description="The title of the news")
    summary: str = Field(description="The summary of the news")
    url: str = Field(description="The URL of the news")
