import os
import io
import base64
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
import numpy as np

load_dotenv()

class GeminiEncoder:
    """
    Wraps Gemini Embedding 2 to produce 3072-dim vectors
    for both text and images in a unified embedding space.
    This is the core innovation over Graph-R1's text-only BGE encoder.
    """

    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = "gemini-embedding-2-preview"
        self.dim = 3072

    def _encode_text(self, text: str) -> np.ndarray:
        response = self.client.models.embed_content(
            model=self.model,
            contents=text
        )
        return np.array(response.embeddings[0].values, dtype=np.float32)

    def _encode_image(self, image_path: str) -> np.ndarray:
        with Image.open(image_path) as img:
            # convert to RGB — strips alpha channel from PNGs
            # without this, RGBA images crash the API
            img = img.convert("RGB")
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            image_bytes = buffer.getvalue()

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        response = self.client.models.embed_content(
            model=self.model,
            contents=types.Content(
                parts=[
                    types.Part(
                        inline_data=types.Blob(
                            mime_type="image/jpeg",
                            data=image_b64
                        )
                    )
                ]
            )
        )
        return np.array(response.embeddings[0].values, dtype=np.float32)

    def encode(self, input) -> np.ndarray:
        """
        Single entry point — pass text string or image path.
        Returns a 3072-dim float32 numpy array either way.
        """
        if isinstance(input, str) and Path(input).is_file() and \
           Path(input).suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
            return self._encode_image(input)
        else:
            return self._encode_text(str(input))

    def encode_batch(self, inputs: list) -> np.ndarray:
        """
        Encode a list of texts and/or image paths.
        1 second sleep between calls — respects free tier rate limit.
        Returns array of shape (len(inputs), 3072)
        """
        embeddings = []
        for i, item in enumerate(inputs):
            embeddings.append(self.encode(item))
            time.sleep(1)
            print(f"Encoded {i+1}/{len(inputs)}", end="\r")
        print(f"Done — {len(inputs)} embeddings built.")
        return np.stack(embeddings)


# =============================================================================
# REUSABLE ENCODER PATTERN — memorize this, not the code
# =============================================================================
#
# STEP 1 — load credentials safely
#   load_dotenv()
#   client = SDKClient(api_key=os.getenv("YOUR_API_KEY"))
#
# STEP 2 — private method per input type (prefix with _)
#   def _encode_text(self, text)  → call text embedding API
#   def _encode_image(self, path) → open image, convert RGB,
#                                   encode to bytes, call image API
#   always return np.array(..., dtype=np.float32)
#   always float32 — FAISS requires it, half the memory of float64
#
# STEP 3 — public encode() as the single decision gate
#   check: is it a string? does the file exist? is it an image extension?
#   yes to all three → _encode_image()
#   anything else   → _encode_text()
#   callers never think about which method — they just call encode()
#
# STEP 4 — public encode_batch() for corpus building
#   loop over inputs, call encode() on each
#   time.sleep(N) between calls — N depends on API rate limit
#   collect results in a list
#   return np.stack(results) → shape (n_inputs, embedding_dim)
#   np.stack not np.array — stack preserves 2D shape correctly
#
# STEP 5 — what changes per project
#   model name string
#   output dim (self.dim)
#   sleep duration (check new API's rate limit)
#   supported image extensions list
#   everything else stays identical
#
# =============================================================================