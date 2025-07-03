from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from colorama import Fore 
from groq import Groq

import json
from typing import List 
from pydantic import BaseModel
from litellm import completion
from generated_prompt import prompt_template
import os
from dotenv import load_dotenv

load_dotenv()


class Record(BaseModel):
    question: str
    answer: str

class Response(BaseModel):
    generated: List[Record]

def llm_call(data: str, num_records: int = 5) -> dict:
    groq_api_key = os.getenv("GENAI_API_KEY")
    client = Groq(api_key=groq_api_key)
    stream = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": prompt_template(data, num_records),
            }
        ],
        stream=True,
        #options={"num_predict": 2000},
        #format=Response.model_json_schema(),
    )
    data = ""
    for x in stream: 
        delta = x['choices'][0]["delta"]["content"]
        if delta is not None: 
            print(Fore.LIGHTBLUE_EX+ delta + Fore.RESET, end="") 
            data += delta 
    return json.loads(data)

if __name__ == "__main__": 
    converter = DocumentConverter()
    doc = converter.convert("pmeendsemnotes.pdf").document
    chunker = HybridChunker()
    chunks = chunker.chunk(dl_doc=doc)

    dataset = {}
    for i, chunk in enumerate(chunks): 
            print(Fore.YELLOW + f"Raw Text:\n{chunk.text[:300]}…" + Fore.RESET)
            enriched_text = chunker.contextualize(chunk=chunk)
            print(Fore.LIGHTMAGENTA_EX + f"Contextualized Tex:\n{enriched_text[:300]}…" + Fore.RESET)

            data = llm_call(
                enriched_text
            )
            if "generated" in data:
                dataset[i] = {"generated": data["generated"], "context": enriched_text}
            else:
                print(Fore.RED + f"\nSkipping chunk {i} due to missing 'generated' key.\n" + Fore.RESET)
    
    with open('tm1_database.json','w') as f: 
        json.dump(dataset, f) 





