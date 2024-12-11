import pandas as pd 
import numpy as np 
import datasets

import requests
from transformers import AutoTokenizer
from bs4 import BeautifulSoup

def parse_wikipedia_text(url):
    try:
        # Fetch the HTML page
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for bad responses

        # Parse the HTML using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract the main content, typically inside <p> tags for Wikipedia
        paragraphs = soup.find_all('p')
        wiki_text = ' '.join([para.get_text() for para in paragraphs])

        # Optionally, you can trim or clean up the text further
        wiki_text = wiki_text.strip()

        return wiki_text

    except requests.exceptions.RequestException as e:
        print(f"Error fetching the Wikipedia page: {e}")
        return None

def add_source_text(sample):
    url = sample['source']
    if '#' in url:
        url_text = None
    else:
        url_text = parse_wikipedia_text(url)
    sample['source_text'] = url_text
    return sample

def count_tokens(s):
    global tokenizer
    return len(tokenizer.encode(s))


if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained('VityaVitalich/Llama3-8b-instruct')

    data = datasets.load_dataset('truthfulqa/truthful_qa', 'generation')
    data = data.filter(lambda x: 'wikipedia' in x['source'])

    data = data.map(add_source_text)
    data = data.filter(lambda x: x['source_text'] is not None)
    data = data.filter(lambda x: count_tokens(x['source_text']) < 4096)


    data.save_to_disk('truthful_qa_with_source')
