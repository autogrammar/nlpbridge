import argparse
import json
import sys
from pathlib import Path
from .common import BridgeError, strict_json
from .config import open_runtime
from .contracts import validate_plan
from .native import load_native, render_plan
from .server import make_server


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="NL → validated plan → DSL; Python + Rust"
    )
    parser.add_argument("--config", default="examples/offline.toml")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    commands.add_parser("index")
    commands.add_parser("status")
    compile_cmd = commands.add_parser("compile")
    compile_cmd.add_argument("text")
    compile_cmd.add_argument("--args")
    compile_cmd.add_argument("--context")
    serve = commands.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--workers", type=int, default=4)
    check = commands.add_parser("validate")
    check.add_argument("plan")
    ns = parser.parse_args(argv)
    runtime = None
    try:
        if ns.command == "doctor":
            result = {
                "python": sys.version.split()[0],
                "native_available": load_native() is not None,
                "native_abi": 1,
            }
        else:
            runtime, stats = open_runtime(ns.config)
            if ns.command == "index":
                result = stats
            elif ns.command == "status":
                result = runtime.health()
            elif ns.command == "compile":
                result = runtime.compile(
                    ns.text,
                    None if ns.context is None else strict_json(ns.context),
                    None if ns.args is None else strict_json(ns.args),
                )
            elif ns.command == "validate":
                plan = strict_json(Path(ns.plan).read_bytes())
                snap = runtime.store.snapshot
                validate_plan(plan, snap.by_uri, snap.revision, runtime.policy)
                result = {
                    "status": "valid",
                    "dsl": render_plan(plan, runtime.store.backend),
                }
            else:
                server = make_server(runtime, ns.host, ns.port, ns.workers)
                print(
                    f"http://{ns.host}:{server.server_address[1]}",
                    file=sys.stderr,
                    flush=True,
                )
                try:
                    server.serve_forever()
                except KeyboardInterrupt:
                    pass
                finally:
                    server.server_close()
                return 0
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (BridgeError, OSError, TypeError, KeyError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    finally:
        if runtime is not None:
            runtime.store.close()
