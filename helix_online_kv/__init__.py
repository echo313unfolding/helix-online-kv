"""Online KV cache compression using calibrate-then-stream codebooks."""

__version__ = "0.3.0"

from .config import OnlineKVConfig
from .codebook import OnlineCodebook
from .vector_codebook import VectorCodebook
from .product_codebook import ProductCodebook
from .layer_state import KVLayerState
from .aging_policy import AgingPolicy

try:
    from .torch_codebook import TorchCodebook
    from .torch_vector_codebook import TorchVectorCodebook
    from .torch_product_codebook import TorchProductCodebook, batched_pq_scores
    from .pq_attention import PQAttentionState
except ImportError:
    pass  # torch not installed
