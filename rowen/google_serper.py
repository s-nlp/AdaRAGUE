# The following code was adapted from https://github.com/hwchase17/langchain/blob/master/langchain/utilities/google_serper.py

"""Util that calls Google Search using the Serper.dev API."""
import asyncio

import aiohttp
import json 
from utils import Sqlite3CacheProvider
import logging
import os

SERPER_API_KEY = os.environ.get('SERPER_API_KEY')
if SERPER_API_KEY is None:
    logging.warning('SERPER_API_KEY is empty')


class GoogleSerperAPIWrapper:
    """Wrapper around the Serper.dev Google Search API."""

    def __init__(self, snippet_cnt=5) -> None:
        self.k = snippet_cnt
        self.gl = "us"
        self.hl = "en"
        self.serper_api_key = SERPER_API_KEY
        self.cache_provider = Sqlite3CacheProvider("google_serper_cache.db")

    async def _google_serper_search_results(
        self, session, search_term: str, gl: str, hl: str
    ) -> dict:
        # Create cache key from request params
        request_params = {
            'search_term': search_term,
            'gl': gl,
            'hl': hl
        }
        cache_key = self.cache_provider.hash_params(request_params)
        
        # Check cache
        cached_result = self.cache_provider.get(cache_key)
        if cached_result:
            return json.loads(cached_result)

        # If not in cache, make the API call
        headers = {
            "X-API-KEY": self.serper_api_key or "",
            "Content-Type": "application/json",
        }
        params = {"q": search_term, "gl": gl, "hl": hl}
        
        async with session.post(
            "https://google.serper.dev/search",
            headers=headers,
            params=params,
            raise_for_status=True,
        ) as response:
            try:
                result = await response.json()
            except Exception as e:
                logging.error(f"Error: {e}")
                return None
            
            # Cache the result
            self.cache_provider.insert(
                cache_key,
                request=request_params,
                response=result,
            )
            
            return result

    def _parse_results(self, results):
        snippets = []

        if isinstance(results, dict):
            if results.get("answerBox"):
                answer_box = results.get("answerBox", {})
                if answer_box.get("answer"):
                    element = {"content": answer_box.get("answer"), "source": "None"}
                    return [element]
                elif answer_box.get("snippet"):
                    element = {
                        "content": answer_box.get("snippet").replace("\n", " "),
                        "source": "None",
                    }
                    return [element]
                elif answer_box.get("snippetHighlighted"):
                    element = {
                        "content": answer_box.get("snippetHighlighted"),
                        "source": "None",
                    }
                    return [element]

            if results.get("knowledgeGraph"):
                kg = results.get("knowledgeGraph", {})
                title = kg.get("title")
                entity_type = kg.get("type")
                if entity_type:
                    element = {"content": f"{title}: {entity_type}", "source": "None"}
                    snippets.append(element)
                description = kg.get("description")
                if description:
                    element = {"content": description, "source": "None"}
                    snippets.append(element)
                for attribute, value in kg.get("attributes", {}).items():
                    element = {"content": f"{attribute}: {value}", "source": "None"}
                    snippets.append(element)

            for result in results["organic"][: self.k]:
                if "snippet" in result:
                    element = {"content": result["snippet"], "source": result["link"]}
                    snippets.append(element)
                for attribute, value in result.get("attributes", {}).items():
                    element = {"content": f"{attribute}: {value}", "source": result["link"]}
                    snippets.append(element)

            if len(snippets) == 0:
                element = {
                    "content": "No good Google Search Result was found",
                    "source": "None",
                }
                return [element]

            # keep only the first k snippets
            snippets = snippets[: int(self.k / 2)]

            return snippets
        

    async def parallel_searches(self, search_queries, gl, hl):
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._google_serper_search_results(session, query, gl, hl)
                for query in search_queries
            ]
            search_results = await asyncio.gather(*tasks, return_exceptions=True)
            return search_results

    async def run(self, queries):
        """Run query through GoogleSearch and parse result."""
        results = await self.parallel_searches(queries, gl=self.gl, hl=self.hl)
        snippets_list = []
        for i in range(len(results)):
            snippets = self._parse_results(results[i])
            if snippets:
                snippets_list.append(snippets)
        return snippets_list


if __name__ == "__main__":
    google_search = GoogleSerperAPIWrapper()
    queries = [
        "Top American film on AFI's list released after 1980",
        "Highest-ranked American movie released after 1980 on AFI's list of 100 greatest films",
        "Top-ranked American film released after 1980 on AFI's list of 100 greatest movies?",
        "AFI's list of 100 greatest American movies released after 1980: top film?",
        "Top-ranked film from AFI's list of 100 greatest American movies released after 1980",
    ]
    search_outputs = asyncio.run(google_search.run(queries))
    retrieved_evidences = [
        query.rstrip("?") + "? " + output["content"]
        for query, result in zip(queries, search_outputs)
        for output in result
    ]
    print(retrieved_evidences)
