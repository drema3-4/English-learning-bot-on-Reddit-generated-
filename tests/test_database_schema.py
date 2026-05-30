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
        "llm_profile_jobs",
        "processing_jobs",
        "review_sessions",
        "user_bot_states",
        "user_learning_profiles",
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


def test_processing_jobs_use_extensible_source_columns() -> None:
    columns = Base.metadata.tables["processing_jobs"].c

    assert "reddit_url" not in columns
    assert {"source_type", "source_ref", "raw_text", "source_metadata"}.issubset(
        columns.keys()
    )
    assert not columns.source_type.nullable
    assert columns.source_ref.nullable
    assert columns.raw_text.nullable


def test_user_learning_profile_schema_is_registered() -> None:
    profile_columns = Base.metadata.tables["user_learning_profiles"].c
    state_columns = Base.metadata.tables["user_bot_states"].c
    llm_profile_columns = Base.metadata.tables["llm_profile_jobs"].c
    processing_columns = Base.metadata.tables["processing_jobs"].c
    extraction_columns = Base.metadata.tables["llm_extraction_jobs"].c

    assert {
        "raw_user_input",
        "cefr_level",
        "goals_summary",
        "profile_json",
        "prompt_profile",
    }.issubset(profile_columns.keys())
    assert {"state", "payload"}.issubset(state_columns.keys())
    assert {"input_text", "prompt_text", "parsed_response"}.issubset(
        llm_profile_columns.keys()
    )
    assert {"profile_id", "profile_snapshot"}.issubset(processing_columns.keys())
    assert {
        "profile_id",
        "profile_snapshot",
        "chunk_index",
        "chunk_count",
    }.issubset(extraction_columns.keys())
