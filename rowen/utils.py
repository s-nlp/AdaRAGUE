import asyncio
import re
import string
import time
import sqlite3 

# import dashscope
import logging
import openai
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
import os
import httpx
from httpx_socks import SyncProxyTransport
import hashlib
import json
from typing import TypedDict, Optional, Dict
from functools import lru_cache
import threading
import traceback

# openai.api_key = "YOUR OPENAI KEY"  # put your openai api key here
# MODEL = "gpt-3.5-turbo"
# BACKEND = 'openai'
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
BACKEND = "vllm" #"openai"
TEMPERATURE = 0.0
PROXY_URI = os.environ.get('PROXY_URI')


class Sqlite3CacheProvider(object):
    CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS cache(
        key string PRIMARY KEY NOT NULL,
        request_params json NOT NULL,
        response json NOT NULL
    );
    """

    def __init__(self, db_path: str = "openai_cache.db"):
        # Allow multi-threading access with proper synchronization
        self.conn: sqlite3.Connection = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()  # Add thread lock
        self.create_table_if_not_exists()

    def get_curr(self) -> sqlite3.Cursor:
        return self.conn.cursor()

    def create_table_if_not_exists(self):
        with self._lock:
            self.get_curr().execute(self.CREATE_TABLE)

    def hash_params(self, params: dict):
        stringified = json.dumps(params).encode("utf-8")
        hashed = hashlib.md5(stringified).hexdigest()
        return hashed

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            res = (
                self.get_curr()
                .execute("SELECT * FROM cache WHERE key= ?", (key,))
                .fetchone()
            )
            return res[-1] if res else None

    def insert(self, key: str, request: dict, response: dict):
        with self._lock:
            self.get_curr().execute(
                "INSERT INTO cache VALUES (?, ?, ?)",
                (
                    key,
                    json.dumps(request),
                    json.dumps(response),
                ),
            )
            self.conn.commit()
        
    def get_size(self) -> int:
        with self._lock:
            res = self.get_curr().execute("SELECT COUNT(*) FROM cache").fetchone()
            return res[0]


def cache_decorator(cache_provider):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Create request params dict from args and kwargs
            request_params = {
                'args': args,
                'kwargs': kwargs
            }
            
            # Get cache key
            key = cache_provider.hash_params(request_params)
            
            try:
                # Check cache
                cached_response = cache_provider.get(key)
                if cached_response:
                    return ChatCompletion.model_validate(json.loads(cached_response))
                
                # Get actual response
                response = func(*args, **kwargs)
                
                # Cache the response atomically
                cache_provider.insert(key, request_params, response.model_dump())
                
                return response
            except sqlite3.Error as e:
                # Handle database errors gracefully
                logging.error(f"Database error occurred: {e}")
                # Fall back to uncached response
                return func(*args, **kwargs)
                
        return wrapper
    return decorator


class LLMClientSingleton:
    _instance_lock = threading.Lock()
    _instance: Dict = {}

    @classmethod
    def get_main_llm_client(cls):
        if BACKEND.lower() == 'vllm':
            return cls.get_local_llm_client()
        elif BACKEND.lower() == 'openai':
            return cls.get_openai_llm_client()
        else:
            raise ValueError(f"Invalid backend: {BACKEND}")

    
    @classmethod
    def get_openai_llm_client(cls):
        # First check without lock
        if 'client_openai' in cls._instance:
            return cls._instance['client_openai']
            
        with cls._instance_lock:
            # Second check with lock
            if 'client' not in cls._instance:
                cache_provider = Sqlite3CacheProvider()
                decorator_fn = cache_decorator(cache_provider)
                
                if PROXY_URI is not None:
                    transport = SyncProxyTransport.from_url(PROXY_URI)
                    http_client = httpx.Client(transport=transport)
                else:
                    http_client = httpx.Client()

                client = OpenAI(
                    http_client=http_client,
                    api_key=os.environ.get("OPENAI_API_KEY"),
                )
                
                client.chat.completions.create = decorator_fn(client.chat.completions.create)
                cls._instance['client_openai'] = client
                
            return cls._instance['client_openai']


    @classmethod
    def get_qwen_llm_client(cls):
        # First check without lock
        if 'client_qwen' in cls._instance:
            return cls._instance['client_qwen']
            
        with cls._instance_lock:
            # Second check with lock
            if 'client_qwen' not in cls._instance:
                cache_provider = Sqlite3CacheProvider('qwen_cache.db')
                decorator_fn = cache_decorator(cache_provider)
                
                client = OpenAI(
                    base_url="http://localhost:8000/v1",
                    api_key="EMPTY",
                )
                client.chat.completions.create = decorator_fn(client.chat.completions.create)
                cls._instance['client_qwen'] = client
            
            return cls._instance['client_qwen']
        

    @classmethod
    def get_local_llm_client(cls):
        # First check without lock
        if 'client_vllm' in cls._instance:
            return cls._instance['client_vllm']
            
        with cls._instance_lock:
            # Second check with lock
            if 'client_vllm' not in cls._instance:
                cache_provider = Sqlite3CacheProvider('localvllm_cache.db')
                decorator_fn = cache_decorator(cache_provider)
                
                client = OpenAI(
                    base_url="http://localhost:8001/v1",
                    api_key="EMPTY",
                )
                client.chat.completions.create = decorator_fn(client.chat.completions.create)
                cls._instance['client_vllm'] = client
            
            return cls._instance['client_vllm']
        
                


def remove_prefix(text: str) -> str:
    result = re.sub(r"^\d+\.\s", "", text)
    return result


def single_run(messages, retry=3, model=MODEL, n=1, temperature=TEMPERATURE):
    for _ in range(retry):
        try:
            client = LLMClientSingleton.get_main_llm_client()
            output = client.chat.completions.create(
                model=model,
                messages=messages,
                n=n,
                temperature=temperature,
            )
            if n == 1:
                return output.choices[0].message.content.strip()
            else:
                return [choice.message.content for choice in output.choices]
        except Exception as e:
            logging.error(e)
            traceback.print_exc()
            time.sleep(20)
    return None


def single_qwen_run(messages, model_name):
    for _ in range(3):
        try:
            client = LLMClientSingleton.get_qwen_llm_client()
            output = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=TEMPERATURE,
            )
            return output.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"single_qwen_run | {str(e)}")
            traceback.print_exc()
            time.sleep(10)
    return None


# async def run_completion_api(prompts):
#     # Make api calls asynchronously
#     async def single_run(prompt, retry=3):
#         for _ in range(retry):
#             try:
#                 client = LLMClientSingleton.get_main_llm_client()
#                 output = client.chat.completions.create(
#                     model="gpt-judge/model/name",
#                     # model="gpt-4o-mini",
#                     prompt=prompt,
#                     temperature=0,
#                     max_tokens=1,
#                     stop=None,
#                     echo=False,
#                     logprobs=2,
#                 )
#                 return output
#             except:
#                 await asyncio.sleep(20)
#         return None

#     responses = [single_run(prompt) for prompt in prompts]
#     return await asyncio.gather(*responses)


async def run_api(messages, model=MODEL, retry=3, temperature=TEMPERATURE):
    # Make api calls asynchronously
    async def single_run(message, model, retry=3, temperature=TEMPERATURE):
        for _ in range(retry):
            try:
                client = LLMClientSingleton.get_main_llm_client()
                output = client.chat.completions.create(
                    model=model,
                    messages=message,
                    temperature=temperature,
                )
                return output.choices[0].message.content.strip()
            except Exception as e:
                logging.error(e)
                await asyncio.sleep(20)
        return None

    responses = [single_run(message, model, retry, temperature) for message in messages]
    return await asyncio.gather(*responses)


def is_supported(generated_answer):
    generated_answer = generated_answer.lower()
    if "true" in generated_answer or "false" in generated_answer:
        if "true" in generated_answer and "false" not in generated_answer:
            is_supported = True
        elif "false" in generated_answer and "true" not in generated_answer:
            is_supported = False
        else:
            is_supported = generated_answer.index("true") > generated_answer.index(
                "false"
            )
    else:
        is_supported = all(
            [
                keyword
                not in generated_answer.lower()
                .translate(str.maketrans("", "", string.punctuation))
                .split()
                for keyword in [
                    "not",
                    "cannot",
                    "unknown",
                    "information",
                ]
            ]
        )
    return is_supported


def is_supported_zh(generated_answer):
    if "是" in generated_answer or "否" in generated_answer:
        if "是" in generated_answer and "否" not in generated_answer:
            is_supported = True
        elif "否" in generated_answer and "是" not in generated_answer:
            is_supported = False
        else:
            is_supported = generated_answer.index("是") > generated_answer.index("否")
    else:
        is_supported = all(
            [
                keyword
                not in generated_answer.lower()
                .translate(str.maketrans("", "", string.punctuation))
                .split()
                for keyword in [
                    "不",
                    "不能",
                    "未知",
                    "信息",
                ]
            ]
        )
    return is_supported
