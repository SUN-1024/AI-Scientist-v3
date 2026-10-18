"""Make a NEW GPU/HPO task using the existing frozen-data construction helpers.

This script does no download. It never opens a prior study's resources. The caller
must supply the official source archive and a CUDA wheel/model verified on the runner.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import shutil
from urllib.parse import urlparse

TEMPLATE = Path(__file__).resolve().parent
EXISTING_TEMPLATE = TEMPLATE.parent / "pathmnist_mil_harness"
BASE_TORCH_VERSION = "2.11.0"
VERIFIED_CUDA_INDEX = "https://download.pytorch.org/whl/cu126"


def validate_runtime_selection(cuda_torch_version, accelerator_model, cuda_wheel_index=VERIFIED_CUDA_INDEX):
    if not isinstance(cuda_torch_version, str) or not re.fullmatch(
        re.escape(BASE_TORCH_VERSION) + r"\+cu[0-9]{3}", cuda_torch_version
    ):
        raise ValueError("supply a verified exact CUDA build of the existing torch 2.11.0 dependency")
    if (not isinstance(accelerator_model, str) or not accelerator_model.strip()
            or accelerator_model in {"none", "UNCONFIGURED"} or len(accelerator_model) > 200):
        raise ValueError("supply the exact CUDA device model observed on the target runner")
    parsed = urlparse(cuda_wheel_index)
    if (parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password
            or parsed.query or parsed.fragment or any(character.isspace() for character in cuda_wheel_index)):
        raise ValueError("CUDA wheel index must be an HTTPS index without credentials or query")
    if parsed.path.rstrip("/").split("/")[-1] != cuda_torch_version.split("+")[-1]:
        raise ValueError("CUDA wheel index and Torch build must use the same CUDA tag")
    return cuda_torch_version, accelerator_model.strip()


def existing_prepare():
    spec = importlib.util.spec_from_file_location("pathmnist_existing_prepare", EXISTING_TEMPLATE / "prepare_data.py")
    if spec is None or spec.loader is None:
        raise ValueError("the existing PathMNIST preparation helper is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_contract(task, *, cuda_torch_version, accelerator_model, helper=None):
    task = Path(task)
    helper = helper or existing_prepare()
    # Reuse the existing artifact builder, including resource hashes and task digest.
    helper.write_contract(task)
    artifacts = json.loads((task / "task_contract.json").read_text())["provenance"]["artifacts"]
    contract = json.loads((TEMPLATE / "task_contract.json").read_text())
    contract["canonical_task"]["sha256"] = helper.digest(task / "task.md")
    for dependency in contract["provenance"]["dependencies"]:
        if dependency["name"] == "torch":
            dependency["version"] = cuda_torch_version
    for plane in ("search", "final"):
        contract["compute"][plane]["accelerator_model"] = accelerator_model
    for path in ("configuration.py", "configuration_contract.json", "shared_helpers/__init__.py",
                 "shared_helpers/evaluator.py", "shared_helpers/protocol.py", "prepare.py", "runtime_setup.json"):
        artifacts.append({"role": "evaluator_helper", "path": path, "sha256": helper.digest(task / path),
                          "source": "new portfolio adapter or frozen existing PathMNIST helper"})
    contract["provenance"]["artifacts"] = artifacts
    (task / "task_contract.json").write_text(json.dumps(contract, indent=2) + "\n")


def prepare(archive, task, *, cuda_torch_version, accelerator_model, cuda_wheel_index=VERIFIED_CUDA_INDEX,
            validation_bags=56, test_bags=32):
    cuda_torch_version, accelerator_model = validate_runtime_selection(cuda_torch_version, accelerator_model,
                                                                      cuda_wheel_index)
    task = Path(task)
    if task.resolve().is_relative_to(TEMPLATE.resolve()) or task.resolve().is_relative_to(EXISTING_TEMPLATE.resolve()):
        raise ValueError("prepare outside the source templates to preserve frozen source code")
    helper = existing_prepare()
    # The original constructor excludes previously used validation/test patch IDs by
    # reconstructing their sampler from raw images. It does not read old holdouts.
    report = helper.prepare(archive, task, validation_bags=validation_bags, test_bags=test_bags)
    for source in TEMPLATE.rglob("*"):
        relative = source.relative_to(TEMPLATE)
        if source.is_file() and "__pycache__" not in relative.parts and source.suffix != ".pyc":
            destination = task / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    for obsolete in ("prepare_data.py", "capture.py"):
        # These copied CPU-case utilities are not entrypoints for the new task.
        (task / obsolete).unlink(missing_ok=True)
    shared = task / "shared_helpers"
    shared.mkdir()
    (shared / "__init__.py").write_text('"""Unmodified scoring helpers from the existing contracted PathMNIST case."""\n')
    for name in ("evaluator.py", "protocol.py"):
        shutil.copyfile(EXISTING_TEMPLATE / name, shared / name)
    lock = (task / "requirements.txt").read_text()
    (task / "requirements.txt").write_text(re.sub(r"(?m)^torch==[^\n]+$", "torch==" + cuda_torch_version, lock))
    setup = {"schema_version": 1, "cuda_torch_version": cuda_torch_version,
             "cuda_wheel_index": cuda_wheel_index,
             "required_image_environment": {"PIP_EXTRA_INDEX_URL": cuda_wheel_index},
             "primary_index_setting": "AGENTIC_TASK_PACKAGE_INDEX",
             "installation_scope": "native managed task runtime; preparation performs no installation",
             "cuda_execution_verified_during_preparation": False}
    (task / "runtime_setup.json").write_text(json.dumps(setup, indent=2) + "\n")
    write_contract(task, cuda_torch_version=cuda_torch_version, accelerator_model=accelerator_model, helper=helper)
    return {**report, "case": "pathmnist_mil_portfolio", "cuda_torch_version": cuda_torch_version,
            "accelerator_model": accelerator_model, "gpu_count_per_candidate": 1,
            "final_evaluation_performed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--cuda-torch-version", required=True)
    parser.add_argument("--accelerator-model", required=True)
    parser.add_argument("--cuda-wheel-index", default=VERIFIED_CUDA_INDEX)
    parser.add_argument("--validation-bags", type=int, default=56)
    parser.add_argument("--test-bags", type=int, default=32)
    arguments = parser.parse_args()
    print(json.dumps(prepare(arguments.archive, arguments.task,
                             cuda_torch_version=arguments.cuda_torch_version,
                             accelerator_model=arguments.accelerator_model,
                             cuda_wheel_index=arguments.cuda_wheel_index,
                             validation_bags=arguments.validation_bags,
                             test_bags=arguments.test_bags), indent=2))
