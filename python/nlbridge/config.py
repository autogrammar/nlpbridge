from pathlib import Path
import tomllib
from .catalog import CatalogStore, load_catalog
from .common import BridgeError
from .contracts import Policy
from .providers import ChatModel, HTTPEmbedder, SentenceEmbedder
from .runtime import Runtime


def open_runtime(config_path):
    path = Path(config_path).resolve()
    with path.open("rb") as f:
        config = tomllib.load(f)

    def relative(p):
        return str((path.parent / p).resolve())

    model_config = dict(config.get("model", {}))
    provider = model_config.pop("provider", "none")
    model = None if provider == "none" else ChatModel(provider=provider, **model_config)
    embedding_config = dict(config.get("embedding", {}))
    provider = embedding_config.pop("provider", "none")
    if provider == "none":
        embedder = None
    elif provider == "sentence_transformers":
        embedder = SentenceEmbedder(**embedding_config)
    else:
        embedder = HTTPEmbedder(provider=provider, **embedding_config)
    p = config.get("policy", {})
    policy = Policy(
        frozenset(p.get("allowed_effects", ["read"])),
        None if "allowed_backends" not in p else frozenset(p["allowed_backends"]),
        None if "allowed_uris" not in p else frozenset(p["allowed_uris"]),
        p.get("max_steps", 8),
    )
    catalog = config.get("catalog", {})
    if "path" not in catalog:
        raise BridgeError("config needs catalog.path")
    source_path = relative(catalog["path"])
    cache_path = relative(catalog.get("database", "../state/catalog.sqlite"))
    if source_path == cache_path:
        raise BridgeError("catalog source and NLBridge cache must be different files")
    store = CatalogStore(cache_path, catalog.get("backend", "auto"))
    try:
        stats = store.sync(load_catalog(source_path), embedder)
        runtime = Runtime(store, model, embedder, policy, **config.get("runtime", {}))
    except Exception:
        store.close()
        raise
    return runtime, stats
