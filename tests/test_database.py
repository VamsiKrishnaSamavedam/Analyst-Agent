import pytest

from ai_agent.services.database import validate_read_only_sql


@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM public.users",
        "WITH active_users AS (SELECT * FROM public.users) SELECT * FROM active_users",
        "SELECT '-- harmless text' AS note",
    ],
)
def test_accepts_read_only_queries(query: str) -> None:
    assert validate_read_only_sql(query)


@pytest.mark.parametrize(
    "query",
    [
        "DELETE FROM public.users",
        "SELECT * FROM public.users; DROP TABLE public.users",
        "UPDATE public.users SET is_active = false",
    ],
)
def test_rejects_unsafe_queries(query: str) -> None:
    with pytest.raises(ValueError):
        validate_read_only_sql(query)
