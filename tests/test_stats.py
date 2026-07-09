from agent.stats import wilson_ci, mcnemar

def test_wilson_basic():
    lo, hi = wilson_ci(8, 10)
    assert 0.4 < lo < 0.8 < hi <= 1.0
    assert wilson_ci(0, 0) == (0.0, 1.0)   # guard: no data -> full interval

def test_mcnemar_symmetric_is_one():
    assert abs(mcnemar(5, 5) - 1.0) < 1e-9

def test_mcnemar_strong_effect_is_small_p():
    assert mcnemar(0, 12) < 0.01           # 12 flips all one way -> significant
