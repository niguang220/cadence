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


def test_empty_question_is_out_of_scope():
    v = classify_intent("")
    assert v.kind == "out_of_scope" and v.reason
