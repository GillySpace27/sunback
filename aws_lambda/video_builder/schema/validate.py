"""A small JSON Schema validator for the public imagery contract (suite SU-9). Standard library only.

validate(instance, schema) returns a list of error strings; an empty list means valid.

Supported keywords: type (a name or a list of names from object, array, string, integer, number,
boolean, null; a bool is never an integer or a number), required, properties, items, pattern
(re.search) and enum. Annotation keywords ($schema, title, description, additionalProperties and
anything else that does not change the answer for the contract) are ignored, so the same files also
load in a full JSON Schema validator. A keyword that would change the answer and is not supported
is an error, never silently skipped.
"""
import re

_TYPES = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}

_UNSUPPORTED = ("$ref", "anyOf", "oneOf", "allOf", "not", "const", "minimum", "maximum", "minLength",
                "maxLength", "minItems", "maxItems", "patternProperties", "if", "then", "else", "dependencies")


def _type_name(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    for name in ("object", "array", "string", "integer", "number"):
        if _TYPES[name](value):
            return name
    return type(value).__name__


def validate(instance, schema, path="$"):
    errors = []
    for keyword in _UNSUPPORTED:
        if keyword in schema:
            errors.append(f"{path}: unsupported schema keyword {keyword!r}")
    expected = schema.get("type")
    if expected is not None:
        names = expected if isinstance(expected, list) else [expected]
        unknown = [n for n in names if n not in _TYPES]
        if unknown:
            errors.append(f"{path}: schema uses unknown type {unknown[0]!r}")
        elif not any(_TYPES[n](instance) for n in names):
            errors.append(f"{path}: expected {' or '.join(names)}, got {_type_name(instance)}")
            return errors
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: {instance!r} is not one of {schema['enum']!r}")
    if "pattern" in schema and isinstance(instance, str) and re.search(schema["pattern"], instance) is None:
        errors.append(f"{path}: {instance!r} does not match {schema['pattern']}")
    if isinstance(instance, dict):
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{path}: missing required key {key!r}")
        for key, sub in schema.get("properties", {}).items():
            if key in instance:
                errors.extend(validate(instance[key], sub, f"{path}.{key}"))
    if isinstance(instance, list) and "items" in schema:
        for i, item in enumerate(instance):
            errors.extend(validate(item, schema["items"], f"{path}[{i}]"))
    return errors
