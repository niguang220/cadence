from agent.plan import Step, Plan, serialize_plan, deserialize_plan, validate_plan

def test_valid_sql_only_plan():
    assert validate_plan(Plan([Step("sql", "count accounts")])).ok

def test_valid_sql_then_python_plan():
    p = Plan([Step("sql", "pull mrr rows"), Step("python", "plot trend")])
    assert validate_plan(p).ok

def test_empty_plan_rejected():
    v = validate_plan(Plan([]))
    assert not v.ok and "empty" in v.reason

def test_bad_kind_rejected():
    v = validate_plan(Plan([Step("shell", "rm -rf")]))
    assert not v.ok and "kind" in v.reason

def test_empty_instruction_rejected():
    v = validate_plan(Plan([Step("sql", "   ")]))
    assert not v.ok and "instruction" in v.reason

def test_python_only_rejected():
    v = validate_plan(Plan([Step("python", "plot")]))
    assert not v.ok and "shape" in v.reason

def test_python_before_sql_rejected():
    v = validate_plan(Plan([Step("python", "plot"), Step("sql", "pull")]))
    assert not v.ok and "shape" in v.reason

def test_two_sql_steps_rejected_in_plan_2():
    v = validate_plan(Plan([Step("sql", "a"), Step("sql", "b")]))
    assert not v.ok and "shape" in v.reason

def test_sql_python_python_rejected_in_plan_2():
    v = validate_plan(Plan([Step("sql", "a"), Step("python", "b"), Step("python", "c")]))
    assert not v.ok and "shape" in v.reason

def test_serialize_round_trip():
    p = Plan([Step("sql", "a"), Step("python", "b")])
    assert deserialize_plan(serialize_plan(p)) == p
