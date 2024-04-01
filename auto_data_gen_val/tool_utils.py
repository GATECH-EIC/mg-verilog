from typing import Optional, Type
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from pydantic import BaseModel, Field
from langchain.retrievers.multi_vector import MultiVectorRetriever
from typing import Any

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)


# You can provide a custom args schema to add descriptions or custom validation
class SCodeRetrieveSchema(BaseModel):
    query: str = Field(description="should be the function name you want to search for")

#TODO: add similarity thresholding
#TODO: multiple doc retrieval
class GlobalCodeRetrieve(BaseTool):
    name = "retrieve_code_function"
    description = "useful for when wantting to look for a function called in a code block to retriveve its summary"
    args_schema: Type[SCodeRetrieveSchema] = SCodeRetrieveSchema
    retriever: Any

    def __init__(self, retriever: Any):
        super(GlobalCodeRetrieve,self).__init__(retriever=retriever)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        doc = self.retriever.vectorstore.similarity_search(query)
        doc_summary = doc[0].metadata["summary"]
        return f"Document summary: {doc_summary}"

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")



