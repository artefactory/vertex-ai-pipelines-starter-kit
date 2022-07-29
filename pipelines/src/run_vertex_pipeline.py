import argparse

from pipelines.src.vertex_pipeline import (
    compile_and_run_pipeline,
    compile_pipeline,
    run_latest_pipeline,
)

FUNCTIONS_MAPPING = {
    "compile": compile_pipeline,
    "compile_and_run": compile_and_run_pipeline,
    "run_latest": run_latest_pipeline,
}

_parser = argparse.ArgumentParser()
_parser.add_argument("--sync", dest="sync", action="store_true")
_parser.add_argument("--async", dest="sync", action="store_false")
_parser.add_argument("--enable-caching", dest="enable_caching", action="store_true")
_parser.add_argument("--disable-caching", dest="enable_caching", action="store_false")
_parser.add_argument("--gcs", dest="local", action="store_false")
_parser.add_argument("--local", dest="local", action="store_true")
_parser.add_argument("--action", choices={"compile", "compile_and_run", "run_latest"})
_parser.set_defaults(sync=False)
_parser.set_defaults(enable_caching=False)
_parser.set_defaults(action="compile_and_run")
_parser.set_defaults(local=True)
if __name__ == "__main__":
    _parsed_args = vars(_parser.parse_args())
    sync = _parsed_args["sync"]
    enable_caching = _parsed_args["enable_caching"]
    action = _parsed_args["action"]
    local = _parsed_args["local"]
    function_to_call = FUNCTIONS_MAPPING[action]
    function_to_call(sync=sync, enable_caching=enable_caching, local=local)
