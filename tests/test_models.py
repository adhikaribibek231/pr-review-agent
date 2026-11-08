import json

import pytest
from pydantic import ValidationError

from pr_agent.models import Category, Finding, ReviewOutput, Severity


def make_finding(**overrides: object) -> Finding:
    fields: dict[str, object] = {
        "filename": "src/app.py",
        "line": 42,
        "severity": "critical",
        "category": "security",
        "message": "User input is interpolated into SQL.",
    }
    fields.update(overrides)
    return Finding.model_validate(fields)


# --------------------------------------------------------------------------
# enums
# --------------------------------------------------------------------------


def test_vocabulary_is_exactly_the_documented_one() -> None:
    assert [s.value for s in Severity] == ["critical", "warning", "info"]
    assert [c.value for c in Category] == ["bug", "security", "style", "convention"]


def test_severity_rank_orders_most_severe_first() -> None:
    ordered = sorted(Severity, key=lambda s: s.rank, reverse=True)
    assert [s.rank for s in Severity] == [0, 1, 2]
    assert ordered == [Severity.INFO, Severity.WARNING, Severity.CRITICAL]


@pytest.mark.parametrize("raw", ["critical", "Critical", "CRITICAL", "  critical\n"])
def test_enum_absorbs_case_and_whitespace_noise(raw: str) -> None:
    assert Severity(raw) is Severity.CRITICAL
    assert make_finding(severity=raw).severity is Severity.CRITICAL


@pytest.mark.parametrize("synonym", ["error", "high", "major", "blocker", ""])
def test_synonyms_are_rejected_not_guessed(synonym: str) -> None:
    with pytest.raises(ValidationError) as error:
        make_finding(severity=synonym)
    assert error.value.errors()[0]["loc"] == ("severity",)


def test_unknown_category_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_finding(category="type-error")


# --------------------------------------------------------------------------
# Finding validation
# --------------------------------------------------------------------------


def test_valid_finding_has_typed_fields() -> None:
    finding = make_finding()
    assert finding.severity is Severity.CRITICAL
    assert finding.category is Category.SECURITY
    assert finding.line == 42


@pytest.mark.parametrize("line", [0, -3])
def test_line_must_be_positive(line: int) -> None:
    with pytest.raises(ValidationError):
        make_finding(line=line)


def test_line_must_be_an_integer() -> None:
    with pytest.raises(ValidationError):
        make_finding(line="forty-two")
    with pytest.raises(ValidationError):
        make_finding(line=4.5)


def test_numeric_string_line_is_coerced() -> None:
    # LLMs sometimes quote numbers; that is formatting, not a wrong answer.
    assert make_finding(line="42").line == 42


@pytest.mark.parametrize("field", ["filename", "message"])
def test_required_text_fields_cannot_be_blank(field: str) -> None:
    for blank in ("", "   \n"):
        with pytest.raises(ValidationError):
            make_finding(**{field: blank})


def test_text_fields_are_stripped() -> None:
    finding = make_finding(filename="  src/app.py\n", message="\n Fix this. ")
    assert finding.filename == "src/app.py"
    assert finding.message == "Fix this."


def test_missing_field_is_rejected() -> None:
    fields = {"filename": "a.py", "line": 1, "severity": "info", "category": "style"}
    with pytest.raises(ValidationError) as error:
        Finding.model_validate(fields)  # no message
    assert error.value.errors()[0]["type"] == "missing"


def test_unexpected_field_is_rejected() -> None:
    with pytest.raises(ValidationError) as error:
        make_finding(confidence=0.9)
    assert error.value.errors()[0]["type"] == "extra_forbidden"


def test_finding_is_immutable_and_hashable() -> None:
    finding = make_finding()
    with pytest.raises(ValidationError):
        finding.line = 7  # pyright: ignore[reportAttributeAccessIssue]
    assert make_finding() == finding
    assert len({finding, make_finding()}) == 1


# --------------------------------------------------------------------------
# ReviewOutput: the shape of a whole LLM reply
# --------------------------------------------------------------------------

REPLY = {
    "findings": [
        {
            "filename": "src/app.py",
            "line": 42,
            "severity": "critical",
            "category": "security",
            "message": "SQL injection.",
        },
        {
            "filename": "src/app.py",
            "line": 50,
            "severity": "Info",
            "category": "style",
            "message": "Prefer a clearer name.",
        },
    ]
}


def test_parses_a_json_reply_into_typed_findings() -> None:
    output = ReviewOutput.model_validate_json(json.dumps(REPLY))
    assert [f.severity for f in output.findings] == [Severity.CRITICAL, Severity.INFO]
    assert output.findings[0].category is Category.SECURITY


def test_an_explicit_empty_list_means_nothing_to_report() -> None:
    assert ReviewOutput.model_validate_json('{"findings": []}').findings == []


def test_omitting_findings_is_an_error_not_a_clean_review() -> None:
    with pytest.raises(ValidationError):
        ReviewOutput.model_validate_json("{}")


def test_a_bare_array_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ReviewOutput.model_validate_json(json.dumps(REPLY["findings"]))


def test_malformed_json_is_a_validation_error_too() -> None:
    # One except clause (ValidationError) covers both bad JSON and bad shape,
    # which is what a "retry once with the error attached" loop needs.
    with pytest.raises(ValidationError):
        ReviewOutput.model_validate_json('{"findings": [')


def test_error_pinpoints_which_finding_and_field_is_wrong() -> None:
    bad = {"findings": [REPLY["findings"][0], {**REPLY["findings"][1], "severity": "high"}]}
    with pytest.raises(ValidationError) as error:
        ReviewOutput.model_validate(bad)
    assert error.value.errors()[0]["loc"] == ("findings", 1, "severity")


def test_one_bad_finding_rejects_the_whole_reply() -> None:
    bad = {"findings": [REPLY["findings"][0], {"filename": "a.py"}]}
    with pytest.raises(ValidationError):
        ReviewOutput.model_validate(bad)


def test_round_trips_through_json() -> None:
    output = ReviewOutput.model_validate(REPLY)
    again = ReviewOutput.model_validate_json(output.model_dump_json())
    assert again == output
    assert json.loads(output.model_dump_json())["findings"][0]["severity"] == "critical"


# --------------------------------------------------------------------------
# the schema we will show the LLM
# --------------------------------------------------------------------------


def test_json_schema_lists_the_allowed_values_and_descriptions() -> None:
    schema = ReviewOutput.model_json_schema()
    defs = schema["$defs"]
    assert defs["Severity"]["enum"] == ["critical", "warning", "info"]
    assert defs["Category"]["enum"] == ["bug", "security", "style", "convention"]
    properties = defs["Finding"]["properties"]
    assert "NEW version" in properties["line"]["description"]
    assert defs["Finding"]["additionalProperties"] is False
    assert schema["required"] == ["findings"]
