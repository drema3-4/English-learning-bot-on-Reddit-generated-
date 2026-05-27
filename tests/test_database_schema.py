from app.db import models  # noqa: F401
from app.db.base import Base


def test_guide_01_tables_are_registered() -> None:
    assert {
        "users",
        "word_lemmas",
        "word_surface_forms",
        "word_usage_notes",
        "phrases",
        "phrase_functions",
        "phrase_examples",
        "rules",
        "rule_examples",
        "llm_extraction_jobs",
        "processing_jobs",
        "review_sessions",
    }.issubset(Base.metadata.tables)


def test_reviewable_entities_default_to_score_one() -> None:
    reviewable_tables = [
        "word_lemmas",
        "word_surface_forms",
        "word_usage_notes",
        "phrases",
        "phrase_functions",
        "phrase_examples",
        "rules",
        "rule_examples",
    ]

    for table_name in reviewable_tables:
        column = Base.metadata.tables[table_name].c.current_score
        assert column.default is not None
        assert column.default.arg == 1
        assert column.server_default is not None


def test_word_lemmas_have_user_scoped_lookup_index() -> None:
    indexes = Base.metadata.tables["word_lemmas"].indexes

    assert any(
        index.name == "ix_word_lemmas_user_id_lemma"
        and [column.name for column in index.columns] == ["user_id", "lemma"]
        for index in indexes
    )
