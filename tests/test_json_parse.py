import pytest

from app.utils.json_parse import JSONParseError, parse_json_array, parse_json_object


def test_parse_direct_json_object() -> None:
    assert parse_json_object('{"words": []}') == {"words": []}


def test_parse_json_from_fenced_block() -> None:
    payload = """
    Here is the result:

    ```json
    {"phrases": [{"text": "show up"}]}
    ```
    """

    assert parse_json_object(payload) == {"phrases": [{"text": "show up"}]}


def test_parse_embedded_json_object() -> None:
    assert parse_json_object('prefix {"rules": []} suffix') == {"rules": []}


def test_parse_direct_json_array() -> None:
    assert parse_json_array('[{"lemma": "show"}]') == [{"lemma": "show"}]


def test_parse_json_array_from_fenced_block() -> None:
    payload = """
    Here is the result:

    ```json
    [{"phrase": "show up"}]
    ```
    """

    assert parse_json_array(payload) == [{"phrase": "show up"}]


def test_parse_embedded_json_array() -> None:
    assert parse_json_array('prefix [{"rule_en": "Use do for questions."}] suffix') == [
        {"rule_en": "Use do for questions."}
    ]


def test_parse_json_array_rejects_non_array() -> None:
    with pytest.raises(JSONParseError):
        parse_json_array("no parseable array here")
