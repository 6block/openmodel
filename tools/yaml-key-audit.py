#!/usr/bin/env python3
"""Config fake-key audit.

YAML silently drops keys the code never declared, so a config written from
memory can LOOK complete while entire features stay off (this exact failure
kept WinningPoSt detection disabled on a production worker for two weeks).
This tool parses the Go config structs' yaml tags into the set of legal key
paths, merges the keys py-inference reads from the same file, and reports
every key in a given YAML file that neither consumer understands.

Usage:
    yaml-key-audit.py <config.go>... -- <file.yaml>...

Exit output lists, per YAML file, the key paths no consumer recognises.
"""
import re
import sys

import yaml

# Keys py-inference reads from the shared sidecar file (its config.py uses
# permissive dict.get(), so the legal set is the set of keys it ever asks for).
PY_KEYS = set(
    "api_port api_token backend bridge_url device_ids enforce_eager engine foc "
    "format gateway gpu_count gpu_memory_utilization grpc inference level "
    "load_balancer logging max_model_len max_tokens_limit mock mock_gpu_count "
    "mode model model_registry multi_gpu multi_instance piece_cid_registry real "
    "simulate_delay supported_models tensor_parallel_size".split()
)
# Free-form maps: any key below these prefixes is data, not schema.
FREE_MAP_PREFIXES = ("foc.mock.model_registry.", "foc.real.piece_cid_registry.")


def parse_go_structs(paths):
    """struct name -> [(yaml key, field type)] from Go source files."""
    structs = {}
    src = ""
    for p in paths:
        src += open(p).read() + "\n"
    for m in re.finditer(r"type\s+(\w+)\s+struct\s*\{(.*?)\n\}", src, re.S):
        name, body = m.group(1), m.group(2)
        fields = []
        for fm in re.finditer(
            r'(\w[\w\d]*)\s+([\[\]\*\w\.]+)\s+`[^`]*yaml:"([^",]+)', body
        ):
            fields.append((fm.group(3), fm.group(2).lstrip("*[]")))
        if fields:
            structs[name] = fields
    return structs


def expand(structs, root):
    """Legal key paths reachable from the root struct (nested structs expanded;
    scalar/map leaves marked with a trailing .* wildcard)."""
    out = set()

    def walk(sname, prefix, seen):
        if sname in seen:
            return
        for key, gtype in structs.get(sname, []):
            path = f"{prefix}{key}"
            out.add(path)
            base = gtype.split(".")[-1]
            if base in structs:
                walk(base, path + ".", seen | {sname})
            else:
                out.add(path + ".*")

    walk(root, "", set())
    return out


def yaml_paths(path):
    data = yaml.safe_load(open(path))
    out = []

    def walk(node, prefix):
        if isinstance(node, dict):
            for k, v in node.items():
                p = f"{prefix}{k}"
                out.append(p)
                walk(v, p + ".")

    walk(data, "")
    return out


if __name__ == "__main__":
    sep = sys.argv.index("--")
    gos, yamls = sys.argv[1:sep], sys.argv[sep + 1 :]
    legal = expand(parse_go_structs(gos), "Config")

    def is_legal(p):
        if p in legal:
            return True
        parts = p.split(".")
        for i in range(len(parts) - 1, 0, -1):
            if ".".join(parts[:i]) + ".*" in legal:
                return True
        if any(p.startswith(fp) for fp in FREE_MAP_PREFIXES):
            return True
        seg = p.split(".")
        if seg[0] in ("inference", "foc") and all(x in PY_KEYS for x in seg):
            return True
        return False

    for y in yamls:
        bad = [p for p in yaml_paths(y) if not is_legal(p)]
        verdict = "OK - no unknown keys" if not bad else f"FAIL - {len(bad)} unknown key(s)"
        print(f"{y}: {verdict}")
        for b in bad:
            print(f"    - {b}")
