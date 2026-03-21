# =============================================================================
# Knowledge Base Builder — Hypergraph Pipeline
# Run once before any queries. Produces FAISS indexes + NetworkX graph.
# =============================================================================

import os
import re
import json
import pickle
import logging
import numpy as np
import faiss
import networkx as nx
from typing import List, Tuple, Dict
from openai import OpenAI
from dotenv import load_dotenv
from graph.encoder import GeminiEncoder

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

encoder = GeminiEncoder()
qwen = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# =============================================================================
# EXTRACTION PROMPT + PARSER
# =============================================================================

EXTRACTION_PROMPT = """\
You are a knowledge graph builder.
Given the text chunk below, extract structured knowledge.

Return ONLY valid JSON. No explanation. No markdown. No code fences.

Format:
{{
  "entities": [
    {{"name": "entity name", "type": "person/place/concept/event"}}
  ],
  "hyperedges": [
    {{
      "fact": "a complete sentence describing a relationship",
      "connects": ["entity name 1", "entity name 2"]
    }}
  ]
}}

Rules:
- Every hyperedge must connect at least 2 entities
- Entity names must exist in the entities list above
- Be concise — entity names 1-4 words max
- Facts must be complete sentences

Text chunk:
{chunk}
"""

def parse_llm_response(response: str) -> dict:
    # step 1 — strip markdown code fences if LLM added them
    response = re.sub(r'```json|```', '', response).strip()
    # step 2 — find JSON object even if text surrounds it
    match = re.search(r'\{.*\}', response, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in LLM response: {response[:200]}")
    return json.loads(match.group())


# =============================================================================
# STEP 1 — CHUNKING
# =============================================================================

def _sliding_window(text: str, chunk_size: int, overlap: int) -> List[str]:

    # step 1 — split entire text into individual words
    words = text.split()
    # "Paris is in France" → ["Paris", "is", "in", "France"]

    # step 2 — calculate how far to jump between chunks
    # if chunk_size=500 and overlap=100, step=400
    # meaning each new chunk starts 400 words after the previous one
    # max(1, ...) ensures step never becomes 0 or negative
    # if it did, range() would loop forever
    step = max(1, chunk_size - overlap)

    # step 3 — build chunks
    chunks = []
    for i in range(0, len(words), step):
        window = words[i : i + chunk_size]  # grab chunk_size words
        if window:                           # skip if empty (end of text)
            chunks.append(" ".join(window)) # rejoin words into string

    return chunks


def chunk_document(
    input_path: str,
    chunk_size: int = 500,
    overlap: int = 100
) -> Tuple[List[str], List[str]]:

    # guard — crash early if file doesn't exist
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    # detect file type from extension
    ext = os.path.splitext(input_path)[-1].lower()
    # "data/sample/doc.PDF" → ".pdf"

    chunks: List[str] = []
    image_paths: List[str] = []

    if ext == ".txt":
        # plain text — just read and chunk
        with open(input_path, "r", encoding="utf-8") as f:
            full_text = f.read()
        chunks = _sliding_window(full_text, chunk_size, overlap)

    elif ext == ".pdf":
        # PDF needs fitz (pymupdf) — handles both text and images
        import fitz
        doc = fitz.open(input_path)
        full_text = ""

        for page_num, page in enumerate(doc):

            # extract text from this page
            full_text += page.get_text()

            # extract images from this page
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]  # image reference ID inside PDF
                base_image = doc.extract_image(xref)

                # save image to disk
                img_path = f"data/sample/img_{page_num}_{img_index}.{base_image['ext']}"
                with open(img_path, "wb") as f:
                    f.write(base_image["image"])

                image_paths.append(img_path)

        # chunk all extracted text
        chunks = _sliding_window(full_text, chunk_size, overlap)

    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # guard — crash if nothing came out
    if not chunks:
        raise ValueError(f"Chunking produced 0 chunks from: {input_path}")

    log.info(f"Chunks: {len(chunks)} | Images: {len(image_paths)}")
    return chunks, image_paths


# =============================================================================
# STEP 2 — ENTITY + HYPEREDGE EXTRACTION (TODO)
# =============================================================================
def extract_entities(
    chunks: List[str],
    image_paths: List[str]
) -> Tuple[List[Dict], List[Dict]]:

    all_entities: List[Dict] = []
    all_hyperedges: List[Dict] = []

    # track seen names to avoid duplicates across chunks
    seen_entity_names: set = set()
    seen_hyperedge_facts: set = set()

    entity_counter = 0
    hyperedge_counter = 0

    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue

        # call Qwen with extraction prompt
        response = qwen.chat.completions.create(
            model="qwen/qwen3-30b-a3b:free",
            messages=[
                {
                    "role": "system",
                    "content": "You are a knowledge graph builder. Return only valid JSON."
                },
                {
                    "role": "user",
                    "content": EXTRACTION_PROMPT.format(chunk=chunk)
                }
            ]
        )

        raw = response.choices[0].message.content

        # defensive parse — never trust LLM output directly
        try:
            parsed = parse_llm_response(raw)
        except ValueError as e:
            log.warning(f"Chunk {i+1} parse failed, skipping: {e}")
            continue

        # process entities
        for e in parsed.get("entities", []):
            if e["name"] not in seen_entity_names:
                all_entities.append({
                    "id": f"e_{entity_counter}",
                    "name": e["name"],
                    "type": e.get("type", "concept")
                })
                seen_entity_names.add(e["name"])
                entity_counter += 1

        # process hyperedges
        for h in parsed.get("hyperedges", []):
            if h["fact"] not in seen_hyperedge_facts:
                all_hyperedges.append({
                    "id": f"h_{hyperedge_counter}",
                    "fact": h["fact"],
                    "connects": h["connects"]
                })
                seen_hyperedge_facts.add(h["fact"])
                hyperedge_counter += 1

        log.info(f"Chunk {i+1}/{len(chunks)} extracted")

    # handle images — each image is one entity + one hyperedge
    for img_path in image_paths:
        response = qwen.chat.completions.create(
            model="qwen/qwen3-30b-a3b:free",
            messages=[
                {
                    "role": "user",
                    "content": f"Describe this image and list the key concepts it represents. Return a single sentence description only.\nImage path: {img_path}"
                }
            ]
        )
        description = response.choices[0].message.content.strip()

        # image itself becomes an entity node
        all_entities.append({
            "id": f"e_{entity_counter}",
            "name": img_path,
            "type": "image"
        })
        entity_counter += 1

        # image description becomes a hyperedge
        all_hyperedges.append({
            "id": f"h_{hyperedge_counter}",
            "fact": description,
            "connects": [img_path]
        })
        hyperedge_counter += 1

    if not all_entities:
        raise ValueError("Extraction produced 0 entities. Check LLM prompt or input.")
    if not all_hyperedges:
        raise ValueError("Extraction produced 0 hyperedges. Check LLM prompt or input.")

    log.info(f"Entities: {len(all_entities)} | Hyperedges: {len(all_hyperedges)}")
    return all_entities, all_hyperedges
# =============================================================================
# STEP 3 — ENCODING (TODO)
# =============================================================================

# =============================================================================
# STEP 4 — BUILD NETWORKX GRAPH (TODO)
# =============================================================================

# =============================================================================
# STEP 5 — BUILD FAISS INDEXES (TODO)
# =============================================================================

# =============================================================================
# STEP 6 — SAVE TO DISK (TODO)
# =============================================================================

# =============================================================================
# MASTER BUILD FUNCTION (TODO)
# =============================================================================