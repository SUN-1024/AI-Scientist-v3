"""Host validation for a candidate's one fixed joint-HPO configuration."""
from __future__ import annotations

import json
import math
from pathlib import Path


def validate_configuration(value, *, contract_path=None):
    path = Path(contract_path) if contract_path else Path(__file__).with_name("configuration_contract.json")
    schema = json.loads(path.read_text())
    rules = schema["parameters"]
    if not isinstance(value, dict) or set(value) != set(rules):
        raise ValueError("configuration must contain exactly the declared parameter names")
    result = {}
    for name, rule in rules.items():
        item = value[name]
        valid_type = type(item) is int if rule["type"] == "integer" else type(item) in (int, float)
        if not valid_type or not math.isfinite(item):
            raise ValueError(f"configuration {name} has invalid numeric type or nonfinite value")
        if "values" in rule and item not in rule["values"]:
            raise ValueError(f"configuration {name} is outside allowed values")
        if "bounds" in rule and not rule["bounds"][0] <= item <= rule["bounds"][1]:
            raise ValueError(f"configuration {name} is outside allowed bounds")
        result[name] = int(item) if rule["type"] == "integer" else float(item)
    return result


def configuration_from_response(response, *, contract_path=None):
    if not isinstance(response, dict) or set(response) != {"configuration"}:
        raise ValueError("get_configuration must return only configuration")
    return validate_configuration(response["configuration"], contract_path=contract_path)


def validate_device_report(value, *, expected_model=None, expected_torch=None):
    required = {"device_type", "cuda_available", "device_count", "logical_device_index",
                "device_name", "device_uuid", "torch_version", "cuda_version"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("candidate device report has invalid fields")
    if (value["device_type"] != "cuda" or value["cuda_available"] is not True
            or type(value["device_count"]) is not int or value["device_count"] != 1
            or type(value["logical_device_index"]) is not int or value["logical_device_index"] != 0):
        raise ValueError("candidate must use exactly one assigned CUDA device as cuda:0")
    for key in ("device_name", "device_uuid", "torch_version", "cuda_version"):
        if not isinstance(value[key], str) or not value[key].strip() or len(value[key]) > 200:
            raise ValueError(f"candidate device report requires bounded nonempty {key}")
    if expected_model is not None and value["device_name"] != expected_model:
        raise ValueError("reported CUDA device model differs from the frozen task contract")
    if expected_torch is not None and value["torch_version"] != expected_torch:
        raise ValueError("reported Torch version differs from the frozen task contract")
    return dict(value)
