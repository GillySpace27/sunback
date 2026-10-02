"""The stdlib JSON Schema subset used by the public imagery contract (SU-9)."""
from aws_lambda.video_builder.schema.validate import validate


def test_valid_instance_gives_an_empty_list():
    schema = {"type": "object", "required": ["a"], "properties": {"a": {"type": "string"}}}
    assert validate({"a": "x"}, schema) == []


def test_type_error_names_the_path():
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
    assert validate({"a": "x"}, schema) == ["$.a: expected integer, got string"]


def test_a_bool_is_not_an_integer_or_a_number():
    assert validate(True, {"type": "integer"}) == ["$: expected integer, got boolean"]
    assert validate(False, {"type": "number"}) == ["$: expected number, got boolean"]
    assert validate(3, {"type": "number"}) == []


def test_type_list_and_null():
    assert validate(None, {"type": ["string", "null"]}) == []
    assert validate(3, {"type": ["string", "null"]}) == ["$: expected string or null, got integer"]


def test_required_keys_are_reported_each():
    schema = {"type": "object", "required": ["a", "b"]}
    assert validate({"a": 1}, schema) == ["$: missing required key 'b'"]
    assert validate({}, schema) == ["$: missing required key 'a'", "$: missing required key 'b'"]


def test_items_paths_carry_the_index():
    schema = {"type": "array", "items": {"type": "object", "required": ["id"]}}
    assert validate([{"id": 1}, {}], schema) == ["$[1]: missing required key 'id'"]


def test_pattern_uses_search_and_reports_the_value():
    schema = {"type": "string", "pattern": "^v/"}
    assert validate("v/171/x", schema) == []
    assert validate("1k/x", schema) == ["$: '1k/x' does not match ^v/"]


def test_enum():
    assert validate("a", {"enum": ["a", "b"]}) == []
    assert validate("c", {"enum": ["a", "b"]}) == ["$: 'c' is not one of ['a', 'b']"]


def test_annotation_keywords_are_ignored_and_behavioural_ones_are_refused():
    schema = {"$schema": "x", "title": "t", "description": "d", "additionalProperties": True, "type": "object"}
    assert validate({"extra": 1}, schema) == []
    assert validate(1, {"anyOf": []}) == ["$: unsupported schema keyword 'anyOf'"]
    assert validate(1, {"$ref": "#/x"}) == ["$: unsupported schema keyword '$ref'"]


def test_nested_error_path():
    schema = {"type": "object", "properties": {"products": {"type": "array", "items": {
        "type": "object", "properties": {"video": {"type": "string"}}}}}}
    assert validate({"products": [{"video": 3}]}, schema) == ["$.products[0].video: expected string, got integer"]
