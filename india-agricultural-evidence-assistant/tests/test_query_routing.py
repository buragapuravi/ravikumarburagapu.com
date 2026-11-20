from app.query_routing import match_event_names, route_query, select_des_district_rows
from app.structured_evidence import load_des_district_records


def test_router_sends_statistics_to_structured_path_and_advisories_to_validity_gate():
    statistics = route_query("What was wheat yield in Ludhiana in 2022-2023?")
    advisory = route_query("Check current rice advisory for North Andaman", "current_advisory")

    assert statistics["path"] == "statistics"
    assert advisory["path"] == "text"
    assert advisory["require_current_advisory"] is True


def test_router_exposes_explicit_agentic_research_path():
    route = route_query("Compare evidence across soil and water management", "agentic")

    assert route["path"] == "agentic"
    assert route["require_current_advisory"] is False


def test_des_question_resolves_supported_dimensions_and_values():
    rows = load_des_district_records()
    result = select_des_district_rows(
        "What was wheat yield in Ludhiana, Punjab during Rabi 2022-2023?",
        rows,
    )

    assert len(result) == 1
    assert result[0]["district"] == "Ludhiana"
    assert result[0]["yield_tonne_per_hectare"] == "4.59"


def test_event_match_requires_a_specific_event_name():
    rows = [
        "ICAR-CIARI Bloomsdale rice field day",
        "ICAR Research Complex pulses rice-fallow field day",
    ]

    assert match_event_names("How many women attended the Bloomsdale rice field day?", rows) == [
        "ICAR-CIARI Bloomsdale rice field day"
    ]
    assert match_event_names("How many farmers attended?", rows) == []
    assert route_query("What did the field-day participants learn?")["path"] == "text"
