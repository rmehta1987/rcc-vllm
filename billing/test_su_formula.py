"""Unit tests for the SU billing formula.

Run with the system pytest (the vllm-probe env has none)::

    /software/python-anaconda-2020.11-el8-x86_64/bin/python -m pytest billing/ -q

Fixtures are the worked examples from the plan
(``glistening-watching-frost.md`` -> "Worked example").
"""

import os

import pytest

import su_formula as su

HERE = os.path.dirname(os.path.abspath(__file__))

# Worked-example throughputs (illustrative; real numbers come from the benchmark).
H200 = dict(w_gpu=3.0, n_gpus=2, prefill_tps=8000.0, decode_tps=2400.0)  # TP=2
A100 = dict(w_gpu=1.0, n_gpus=4, prefill_tps=6000.0, decode_tps=1800.0)  # TP=4
REQ = dict(t_in=2000, t_out=500)


# --------------------------------------------------------------------------- #
# Per-request charge -- worked examples
# --------------------------------------------------------------------------- #
def test_worked_example_h200_token_su():
    val = su.su_for_request(**REQ, **H200)
    assert val == pytest.approx(0.00076, abs=5e-6)


def test_worked_example_a100_token_su():
    val = su.su_for_request(**REQ, **A100)
    assert val == pytest.approx(0.00068, abs=5e-6)


def test_per_1k_form_matches_gpu_time_form():
    """The runtime rate-table form must equal the GPU-time form exactly."""
    for tier in (H200, A100):
        spk_in = su.su_per_1k_tokens(tier["w_gpu"], tier["n_gpus"], tier["prefill_tps"])
        spk_out = su.su_per_1k_tokens(tier["w_gpu"], tier["n_gpus"], tier["decode_tps"])
        from_rates = su.su_for_request_from_rates(REQ["t_in"], REQ["t_out"], spk_in, spk_out)
        gpu_time = su.su_for_request(**REQ, **tier)
        assert from_rates == pytest.approx(gpu_time, rel=1e-12)


def test_alpha_empirical():
    assert su.alpha_empirical(8000.0, 2400.0) == pytest.approx(8000.0 / 2400.0)
    # decode is slower than prefill on every real tier -> alpha > 1
    assert su.alpha_empirical(H200["prefill_tps"], H200["decode_tps"]) > 1.0


def test_output_costs_more_than_input_per_token():
    """Asymmetry falls out of the physics: an output token costs alpha x an input one."""
    in_only = su.su_for_request(t_in=1000, t_out=0, **H200)
    out_only = su.su_for_request(t_in=0, t_out=1000, **H200)
    assert out_only > in_only
    assert out_only / in_only == pytest.approx(su.alpha_empirical(8000.0, 2400.0))


# --------------------------------------------------------------------------- #
# Reservation floor + session max()
# --------------------------------------------------------------------------- #
def test_worked_example_floors():
    assert su.reservation_floor_su(3.0, 2, 2.0) == pytest.approx(12.0)  # H200
    assert su.reservation_floor_su(1.0, 4, 2.0) == pytest.approx(8.0)   # A100


def test_floor_dominates_on_exclusive_node():
    """A single request's token cost is dwarfed by the multi-hour floor."""
    token_su = su.su_for_request(**REQ, **H200)  # ~0.00076
    charge = su.su_for_session(token_su, H200["w_gpu"], H200["n_gpus"], 2.0)
    assert charge.basis == "floor"
    assert charge.billed_su == pytest.approx(12.0)


def test_token_term_wins_when_work_exceeds_floor():
    """If summed token SU exceeds the floor (sustained high concurrency), bill tokens."""
    big_token_su = 20.0
    charge = su.su_for_session(big_token_su, H200["w_gpu"], H200["n_gpus"], 2.0)
    assert charge.basis == "tokens"
    assert charge.billed_su == pytest.approx(20.0)


def test_floor_disabled_bills_tokens():
    charge = su.su_for_session(0.001, H200["w_gpu"], H200["n_gpus"], 2.0, floor_enabled=False)
    assert charge.billed_su == pytest.approx(0.001)
    assert "floor disabled" in charge.basis


def test_floor_fixes_the_perversion():
    """The floor ordering H200 > A100 holds by construction (pure w_gpu*N, no throughput).

    This is the specific bug the new formula fixes: the old flat rate had no
    floor and put throughput in a charge-reducing position, so H200 was
    strictly cheapest. Here H200 always costs more to *hold*.
    """
    h200_floor = su.reservation_floor_su(3.0, 2, 1.0)  # 6.0
    a100_floor = su.reservation_floor_su(1.0, 4, 1.0)  # 4.0
    a40_floor = su.reservation_floor_su(0.5, 1, 1.0)   # 0.5
    assert h200_floor > a100_floor > a40_floor


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #
def test_failed_requests_not_billed():
    reqs = [
        {"prompt_tokens": 2000, "completion_tokens": 500, "success": True},
        {"prompt_tokens": 100, "completion_tokens": 0, "success": False},  # 5xx
        {"prompt_tokens": 2000, "completion_tokens": 500},  # success default True
    ]
    total, n = su.sum_request_su(reqs, **H200, bill_failed=False)
    assert n == 2
    expected = 2 * su.su_for_request(2000, 500, **H200)
    assert total == pytest.approx(expected)


def test_bill_failed_toggle():
    reqs = [{"prompt_tokens": 100, "completion_tokens": 50, "success": False}]
    total, n = su.sum_request_su(reqs, **H200, bill_failed=True)
    assert n == 1
    assert total > 0


def test_prefix_cache_bills_full_prompt():
    """No cache discount: prompt_tokens from usage is billed in full."""
    cached = {"prompt_tokens": 5000, "completion_tokens": 100, "success": True}
    total, _ = su.sum_request_su([cached], **H200)
    assert total == pytest.approx(su.su_for_request(5000, 100, **H200))


def test_zero_tokens_is_zero_su():
    assert su.su_for_request(0, 0, **H200) == 0.0


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_nonpositive_throughput_raises(bad):
    with pytest.raises(ValueError):
        su.su_per_1k_tokens(1.0, 1, bad)
    with pytest.raises(ValueError):
        su.su_for_request(100, 100, 1.0, 1, prefill_tps=bad, decode_tps=100.0)


def test_negative_tokens_raise():
    with pytest.raises(ValueError):
        su.su_for_request(-1, 0, **H200)


# --------------------------------------------------------------------------- #
# Policy + rate-table loaders (need yaml -- skip if unavailable)
# --------------------------------------------------------------------------- #
def test_policy_loads_and_weights():
    pytest.importorskip("yaml")
    pol = su.load_policy(os.path.join(HERE, "billing_policy.yaml"))
    assert pol.su_per_a100_gpu_hour == 1.0
    assert pol.weight("a100") == 1.0
    assert pol.weight("h200") == 3.0
    assert pol.weight("a40") == 0.5
    assert pol.floor_enabled is True
    assert pol.bill_failed is False
    assert pol.prefix_cache_discount is False


def test_policy_tier_normalization():
    pytest.importorskip("yaml")
    pol = su.load_policy(os.path.join(HERE, "billing_policy.yaml"))
    # real nvidia-smi style names must resolve to a tier
    assert pol.weight("NVIDIA A100-SXM4-40GB") == 1.0
    assert pol.weight("NVIDIA H200") == 3.0
    assert pol.weight("H100") == 2.0


def test_excluded_tier_raises():
    pytest.importorskip("yaml")
    pol = su.load_policy(os.path.join(HERE, "billing_policy.yaml"))
    with pytest.raises(KeyError):
        pol.weight("v100")
    with pytest.raises(KeyError):
        pol.weight("rtx6000")


def test_unknown_tier_raises():
    pytest.importorskip("yaml")
    pol = su.load_policy(os.path.join(HERE, "billing_policy.yaml"))
    with pytest.raises(KeyError):
        pol.weight("mi300x")


def test_rate_table_loads_and_misses_fall_back():
    rt = su.load_rate_table(os.path.join(HERE, "rate_table.json"))
    assert "records" in rt
    assert isinstance(rt["records"], list)
    # unbenchmarked (model, tier, TP) -> lookup returns None (metering falls back to floor)
    assert su.rate_record(rt, "no_such_model", "a100", 4) is None
    assert su.rate_record(rt, "qwen2.5_72B", "a100", 99) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
