# Frozen source excerpt: tests/test_agent_budget.py:40-62
def test_restart_counts_dispatched_calls_even_if_process_did_not_record_result(saved_case):
    store, case_id = saved_case
    store.add_event(case_id, {'phase': 'model', 'name': 'investigator', 'status': 'started'})
    restarted = InvestigationBudget(store, case_id, max_model_calls=1)
    with pytest.raises(BudgetExceeded):
        BudgetMiddleware(restarted).wrap_model_call(None, response)


def test_parallel_tool_reservations_never_exceed_shared_limit(saved_case):
    store, case_id = saved_case
    budget = InvestigationBudget(store, case_id, max_tool_calls=3)

    def execute(_):
        try:
            with budget.tool_call('read_case_evidence'):
                return True
        except BudgetExceeded:
            return False

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(execute, range(12)))
    assert sum(results) == 3
    assert sum(e['phase'] == 'tool' for e in store.events(case_id)) == 3
