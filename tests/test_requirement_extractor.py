from src.requirement_extractor import extract_requirements


def test_quantity_detection():
    reqs = extract_requirements("Include at least three real-world examples.")
    quantities = [r for r in reqs if r.kind == "quantity"]
    assert quantities, "expected a quantity requirement to be detected"
    assert "three" in quantities[0].numbers
    assert quantities[0].critical is True


def test_negative_detection():
    reqs = extract_requirements("Avoid unnecessary mathematical details.")
    assert reqs[0].kind == "negative"
    assert reqs[0].critical is True


def test_format_detection_table_and_headings():
    reqs = extract_requirements(
        "Organize the answer using clear headings and a comparison table."
    )
    fmt = [r for r in reqs if r.kind == "format"]
    assert fmt, "expected a format requirement to be detected"
    assert "comparison table" in fmt[0].terms or "table" in fmt[0].terms
    assert "headings" in fmt[0].terms or "heading" in fmt[0].terms


def test_persona_sentence_is_not_critical():
    reqs = extract_requirements(
        "You are an experienced data scientist and machine learning mentor."
    )
    assert reqs[0].kind == "persona"
    assert reqs[0].critical is False


def test_inclusion_verb_detection():
    reqs = extract_requirements("Explain supervised learning to a beginner.")
    assert reqs[0].kind == "inclusion"
    assert reqs[0].critical is True


def test_exactly_n_quantity():
    reqs = extract_requirements("Provide exactly one short code example for each concept.")
    quantities = [r for r in reqs if r.kind == "quantity"]
    assert quantities
    assert "one" in quantities[0].numbers


def test_json_format_detection():
    reqs = extract_requirements("Output the result as JSON.")
    fmt = [r for r in reqs if r.kind == "format"]
    assert fmt
    assert "json" in fmt[0].terms