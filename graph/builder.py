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
# STEP 2 — ENTITY + HYPEREDGE EXTRACTION
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
# STEP 3 — ENCODING
# =============================================================================
def encode_entities(entities: List[Dict]) -> np.ndarray:
    if not entities:
        raise ValueError("Cannot encode: entity list is empty.")
    
    # name + type for richer semantic representation
    texts = [f"{e['name']} {e.get('type', '')}" for e in entities]
    embeddings = encoder.encode_batch(texts)
    return embeddings.astype(np.float32)


def encode_hyperedges(hyperedges: List[Dict]) -> np.ndarray:
    if not hyperedges:
        raise ValueError("Cannot encode: hyperedge list is empty.")
    
    # encode the full fact sentence
    texts = [h["fact"] for h in hyperedges]
    embeddings = encoder.encode_batch(texts)
    return embeddings.astype(np.float32)
# =============================================================================
# STEP 4 — BUILD NETWORKX GRAPH
# =============================================================================
def build_graph(
    entities: List[Dict],
    hyperedges: List[Dict]
) -> nx.Graph:

    G = nx.Graph()

    # add all entity nodes first
    for e in entities:
        G.add_node(
            e["name"],
            type="entity",
            entity_id=e["id"],
            entity_type=e.get("type", "unknown")
        )

    # add hyperedge nodes + connect to entities
    entity_name_set = {e["name"] for e in entities}
    skipped = 0

    for h in hyperedges:
        # hyperedge ID is its node key in the graph
        G.add_node(
            h["id"],
            type="hyperedge",
            fact=h["fact"]
        )
        # connect hyperedge to every entity it mentions
        for entity_name in h["connects"]:
            if entity_name in entity_name_set:
                G.add_edge(h["id"], entity_name)
            else:
                skipped += 1
                log.warning(f"Skipped: '{entity_name}' not in entity set")

    if skipped:
        log.warning(f"Total skipped connections: {skipped}")

    log.info(f"Graph: {G.number_of_nodes()} nodes | {G.number_of_edges()} edges")
    return G
# =============================================================================
# STEP 5 — BUILD FAISS INDEXES
# =============================================================================
def build_faiss_indexes(
    entity_embeddings: np.ndarray,
    hyperedge_embeddings: np.ndarray
) -> Tuple[faiss.Index, faiss.Index]:

    # validate inputs are 2D
    if entity_embeddings.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {entity_embeddings.shape}")
    if hyperedge_embeddings.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {hyperedge_embeddings.shape}")

    # validate both come from same encoder — dims must match
    if entity_embeddings.shape[1] != hyperedge_embeddings.shape[1]:
        raise ValueError(
            f"Dimension mismatch: {entity_embeddings.shape[1]} vs "
            f"{hyperedge_embeddings.shape[1]}"
        )

    dim = entity_embeddings.shape[1]

    # IndexFlatIP = exact search using inner product (cosine similarity)
    # row 0 in index = entities[0], row 1 = entities[1], order must never change
    index_entity = faiss.IndexFlatIP(dim)
    index_entity.add(entity_embeddings)

    index_hyperedge = faiss.IndexFlatIP(dim)
    index_hyperedge.add(hyperedge_embeddings)

    log.info(f"FAISS entity index: {index_entity.ntotal} vectors")
    log.info(f"FAISS hyperedge index: {index_hyperedge.ntotal} vectors")
    return index_entity, index_hyperedge
# =============================================================================
# STEP 6 — SAVE TO DISK
# =============================================================================
def save(
    G: nx.Graph,
    index_entity: faiss.Index,
    index_hyperedge: faiss.Index,
    entities: List[Dict],
    hyperedges: List[Dict],
    save_dir: str = "artifacts"
) -> None:

    os.makedirs(save_dir, exist_ok=True)

    # save FAISS indexes in binary format
    faiss.write_index(index_entity,
        os.path.join(save_dir, "index_entity.bin"))
    faiss.write_index(index_hyperedge,
        os.path.join(save_dir, "index_hyperedge.bin"))

    # save metadata — maps integer ID → node content
    metadata = {
        "entities": entities,
        "hyperedges": hyperedges,
        "entity_count": len(entities),
        "hyperedge_count": len(hyperedges)
    }
    with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # save NetworkX graph — pickle format
    # nx.write_gpickle deprecated in 3.x — use pickle directly
    with open(os.path.join(save_dir, "graph.gpickle"), "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    log.info(f"Saved to: {os.path.abspath(save_dir)}/")
    log.info(f"  index_entity.bin ({index_entity.ntotal} vectors)")
    log.info(f"  index_hyperedge.bin ({index_hyperedge.ntotal} vectors)")
    log.info(f"  metadata.json ({len(entities)} entities, {len(hyperedges)} hyperedges)")
    log.info(f"  graph.gpickle ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
# =============================================================================
# MASTER BUILD FUNCTION
# =============================================================================
def build(input_path: str, save_dir: str = "artifacts") -> None:
    """
    End-to-end pipeline. Only function main.py ever calls.
    """
    log.info("=" * 50)
    log.info("Starting hypergraph build pipeline")
    log.info(f"Input: {input_path}")
    log.info("=" * 50)

    try:
        log.info("[Step 1/6] Chunking document...")
        chunks, image_paths = chunk_document(input_path)

        log.info("[Step 2/6] Extracting entities and hyperedges...")
        entities, hyperedges = extract_entities(chunks, image_paths)

        log.info("[Step 3/6] Encoding entities and hyperedges...")
        entity_embeddings = encode_entities(entities)
        hyperedge_embeddings = encode_hyperedges(hyperedges)

        log.info("[Step 4/6] Building NetworkX graph...")
        G = build_graph(entities, hyperedges)

        log.info("[Step 5/6] Building FAISS indexes...")
        index_e, index_h = build_faiss_indexes(
            entity_embeddings,
            hyperedge_embeddings
        )

        log.info("[Step 6/6] Saving artifacts to disk...")
        save(G, index_e, index_h, entities, hyperedges, save_dir)

        log.info("=" * 50)
        log.info("Build complete.")
        log.info(f"  Entities:   {len(entities)}")
        log.info(f"  Hyperedges: {len(hyperedges)}")
        log.info(f"  Artifacts:  {os.path.abspath(save_dir)}/")
        log.info("=" * 50)

    except NotImplementedError as e:
        log.error(f"Not implemented: {e}")
        raise
    except FileNotFoundError as e:
        log.error(f"File not found: {e}")
        raise
    except ValueError as e:
        log.error(f"Bad data: {e}")
        raise
    except Exception as e:
        log.error(f"Build failed at: {type(e).__name__}: {e}")
        raise