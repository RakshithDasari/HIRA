# =============================================================================
# Incremental Knowledge Base Updater
# Adds new data to existing artifacts without rebuilding from scratch.
# React virtual DOM pattern applied to knowledge graphs.
# Fixes Graph-R1's static hypergraph limitation.
# =============================================================================

import os
import json
import pickle
import logging
import faiss
import networkx as nx
from dotenv import load_dotenv
from graph.encoder import GeminiEncoder
from graph.builder import (
    chunk_document,
    extract_entities,
    encode_entities,
    encode_hyperedges,
)

load_dotenv()
log = logging.getLogger(__name__)

encoder = GeminiEncoder()


class Updater:
    """
    Incrementally updates existing artifacts with new documents or images.
    Only adds genuinely new entities and hyperedges — never rebuilds.
    React virtual DOM pattern applied to knowledge graphs.
    Fixes Graph-R1's static hypergraph limitation.

    Supports three update modes:
      1. Document only  — update("doc.txt")
      2. Document + images — update("doc.txt", image_paths=["img1.png"])
      3. Images only    — update(image_paths=["img1.png", "img2.png"])
    """

    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = artifacts_dir
        self._load_artifacts()

    def _load_artifacts(self):
        # load FAISS indexes
        self.index_entity = faiss.read_index(
            os.path.join(self.artifacts_dir, "index_entity.bin")
        )
        self.index_hyperedge = faiss.read_index(
            os.path.join(self.artifacts_dir, "index_hyperedge.bin")
        )

        # load metadata
        with open(
            os.path.join(self.artifacts_dir, "metadata.json"),
            encoding="utf-8"
        ) as f:
            metadata = json.load(f)

        self.entities = metadata["entities"]
        self.hyperedges = metadata["hyperedges"]

        # load graph
        with open(
            os.path.join(self.artifacts_dir, "graph.gpickle"),
            "rb"
        ) as f:
            self.G = pickle.load(f)

        log.info(
            f"Updater loaded: {len(self.entities)} entities, "
            f"{len(self.hyperedges)} hyperedges"
        )

    def _compute_diff(
        self,
        new_entities: list,
        new_hyperedges: list
    ) -> tuple:
        """
        Compares new entities/hyperedges against existing ones.
        Returns only the genuinely new ones — the diff.
        O(1) set-based lookup — same pattern as builder.py.
        """
        existing_entity_names = {e["name"] for e in self.entities}
        existing_hyperedge_facts = {h["fact"] for h in self.hyperedges}

        fresh_entities = [
            e for e in new_entities
            if e["name"] not in existing_entity_names
        ]
        fresh_hyperedges = [
            h for h in new_hyperedges
            if h["fact"] not in existing_hyperedge_facts
        ]

        log.info(
            f"Diff: {len(fresh_entities)} new entities, "
            f"{len(fresh_hyperedges)} new hyperedges "
            f"(skipped {len(new_entities) - len(fresh_entities)} duplicates)"
        )
        return fresh_entities, fresh_hyperedges

    def _update_faiss(
        self,
        fresh_entities: list,
        fresh_hyperedges: list
    ) -> None:
        """
        Appends new vectors to existing FAISS indexes.
        Never rebuilds — only appends.
        """
        if fresh_entities:
            new_entity_embeddings = encode_entities(fresh_entities)
            self.index_entity.add(new_entity_embeddings)

        if fresh_hyperedges:
            new_hyperedge_embeddings = encode_hyperedges(fresh_hyperedges)
            self.index_hyperedge.add(new_hyperedge_embeddings)

    def _update_graph(
        self,
        fresh_entities: list,
        fresh_hyperedges: list
    ) -> None:
        """
        Appends new nodes and edges to existing NetworkX graph.
        """
        entity_name_set = {e["name"] for e in self.entities + fresh_entities}

        for e in fresh_entities:
            self.G.add_node(
                e["name"],
                type="entity",
                entity_id=e["id"],
                entity_type=e.get("type", "unknown")
            )

        for h in fresh_hyperedges:
            self.G.add_node(
                h["id"],
                type="hyperedge",
                fact=h["fact"]
            )
            for entity_name in h["connects"]:
                if entity_name in entity_name_set:
                    self.G.add_edge(h["id"], entity_name)

    def _save_artifacts(self) -> None:
        """
        Persists updated artifacts back to disk.
        Overwrites existing files with updated versions.
        """
        faiss.write_index(
            self.index_entity,
            os.path.join(self.artifacts_dir, "index_entity.bin")
        )
        faiss.write_index(
            self.index_hyperedge,
            os.path.join(self.artifacts_dir, "index_hyperedge.bin")
        )

        metadata = {
            "entities": self.entities,
            "hyperedges": self.hyperedges,
            "entity_count": len(self.entities),
            "hyperedge_count": len(self.hyperedges)
        }
        with open(
            os.path.join(self.artifacts_dir, "metadata.json"),
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        with open(
            os.path.join(self.artifacts_dir, "graph.gpickle"),
            "wb"
        ) as f:
            pickle.dump(self.G, f, protocol=pickle.HIGHEST_PROTOCOL)

        log.info(
            f"Artifacts updated: {len(self.entities)} entities, "
            f"{len(self.hyperedges)} hyperedges"
        )

    def update(
        self,
        input_path: str = None,
        image_paths: list = None
    ) -> dict:
        """
        Main entry point. Supports three modes:
          1. Document only:        update("doc.txt")
          2. Document + images:    update("doc.txt", image_paths=["img.png"])
          3. Images only:          update(image_paths=["img1.png", "img2.png"])

        The fix: chunk_document() only accepts .txt and .pdf.
        Images bypass chunk_document entirely — they go directly into
        extract_entities() as image_paths with empty chunks=[].
        This is the correct architectural fix — same pattern builder.py
        uses when a PDF has embedded images.
        """
        if image_paths is None:
            image_paths = []

        # validate — need at least one input
        if not input_path and not image_paths:
            raise ValueError("Provide input_path, image_paths, or both.")

        log.info("=" * 50)
        log.info("Starting incremental update")
        if input_path:
            log.info(f"  Document: {input_path}")
        if image_paths:
            log.info(f"  Images:   {image_paths}")
        log.info("=" * 50)

        chunks = []
        extracted_image_paths = []

        # step 1 — chunk document if provided
        if input_path:
            log.info("[Step 1] Chunking document...")
            # chunk_document handles .txt and .pdf only — correct
            doc_chunks, doc_images = chunk_document(input_path)
            chunks.extend(doc_chunks)
            extracted_image_paths.extend(doc_images)
        else:
            log.info("[Step 1] No document — image-only update, skipping chunking")

        # merge any explicitly passed image paths
        # (images uploaded via UI separately from the document)
        for ip in image_paths:
            if ip not in extracted_image_paths:
                extracted_image_paths.append(ip)

        if extracted_image_paths:
            log.info(f"[Step 1] Total images to process: {len(extracted_image_paths)}")

        # step 2 — extract entities and hyperedges
        # chunks=[] is valid — extract_entities handles images in image_paths
        # even when there are no text chunks
        log.info("[Step 2] Extracting entities and hyperedges...")
        new_entities, new_hyperedges = extract_entities(chunks, extracted_image_paths)

        # step 3 — compute diff
        log.info("[Step 3] Computing diff against existing knowledge base...")
        fresh_entities, fresh_hyperedges = self._compute_diff(
            new_entities,
            new_hyperedges
        )

        if not fresh_entities and not fresh_hyperedges:
            log.info("No new knowledge found — artifacts unchanged")
            return {"added_entities": 0, "added_hyperedges": 0}

        # step 4 — update FAISS
        log.info(f"[Step 4] Encoding {len(fresh_entities)} new entities...")
        self._update_faiss(fresh_entities, fresh_hyperedges)

        # step 5 — update NetworkX graph
        log.info("[Step 5] Patching NetworkX graph...")
        self._update_graph(fresh_entities, fresh_hyperedges)

        # step 6 — extend metadata
        self.entities.extend(fresh_entities)
        self.hyperedges.extend(fresh_hyperedges)

        # step 7 — save everything
        log.info("[Step 7] Saving artifacts to disk...")
        self._save_artifacts()

        log.info(
            f"Update complete: +{len(fresh_entities)} entities, "
            f"+{len(fresh_hyperedges)} hyperedges"
        )
        log.info("=" * 50)

        return {
            "added_entities": len(fresh_entities),
            "added_hyperedges": len(fresh_hyperedges)
        }


# =============================================================================
# REUSABLE UPDATER PATTERN — for future projects
# =============================================================================
#
# 1. load existing artifacts from disk
# 2. chunk document (if provided) — chunk_document handles .txt and .pdf only
# 3. images bypass chunking — passed directly to extract_entities as image_paths
# 4. extract entities from chunks + image_paths
# 5. compute diff — compare new vs existing by semantic identity
#    entities   → compare by name
#    hyperedges → compare by fact string
# 6. update FAISS — append new vectors only
# 7. update NetworkX — add new nodes and edges only
# 8. extend metadata lists
# 9. save artifacts back to disk
#
# key principle: never rebuild, only append
# images bypass chunk_document — this is architecturally correct
# chunk_document's job is splitting text, not handling images
# images are entities by themselves, not chunks
# =============================================================================