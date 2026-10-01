"""Run from the project root: python examples/in_process.py"""

from nlbridge.config import open_runtime
from nlbridge.executor import execute

runtime, _ = open_runtime("examples/offline.toml")
try:
    result = runtime.compile(
        "proc://demo/text/word-count/v1", args={"text": "Zażółć gęślą jaźń"}
    )
    print(result["dsl"])
    bindings = {
        "proc://demo/text/word-count/v1": lambda text: {"count": len(text.split())}
    }
    print(execute(result["plan"], runtime.store.snapshot, bindings, runtime.policy))
finally:
    runtime.store.close()
