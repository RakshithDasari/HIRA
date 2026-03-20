from graph.encoder import GeminiEncoder
import numpy as np


encoder = GeminiEncoder()

vec = encoder.encode("Paris is the capital of France")
print(f"Text vector shape: {vec.shape}")
assert vec.shape == (3072,), "Wrong shape"
assert vec.dtype == np.float32, "Wrong dtype"
assert vec.sum() != 0, "Empty vector"

print("All tests passed.")