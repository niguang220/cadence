from agent.intent import classify_intent


def test_greeting_is_out_of_scope():
    v = classify_intent("hi there, how are you?")
    assert v.kind == "out_of_scope" and v.reason


def test_meta_question_is_out_of_scope():
    assert classify_intent("who are you and what can you do?").kind == "out_of_scope"


def test_data_question_is_in_scope():
    # conservative default: anything that isn't an obvious greeting/meta is 'data'
    assert classify_intent("how many customers signed up last month?").kind == "data"


def test_ambiguous_but_data_stays_data():
    assert classify_intent("best customers").kind == "data"


def test_polite_prefixed_data_question_stays_in_scope():
    # regression: a greeting/politeness PREFIX on a real data question must NOT be refused.
    assert classify_intent("Hi, how many accounts do we have?").kind == "data"
    assert classify_intent("Thanks, show this month's MRR").kind == "data"


def test_empty_question_is_out_of_scope():
    v = classify_intent("")
    assert v.kind == "out_of_scope" and v.reason
