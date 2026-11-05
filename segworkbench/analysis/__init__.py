try:
    from .gmm_segmenter import GMMSegmenter

    __all__ = ["GMMSegmenter"]
except ImportError:  # pragma: no cover
    __all__ = []
