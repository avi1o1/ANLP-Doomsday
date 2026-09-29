import itertools

import pytest

from scripts.shapley_attribution import SWITCHES, configuration, shapley


def lattice(function):
    return {configuration({s for s, bit in zip(SWITCHES, bits) if bit}): function(dict(zip(SWITCHES, bits)))
            for bits in itertools.product((0, 1), repeat=3)}


def test_values_sum_to_the_all_on_gain():
    quality = lattice(lambda on: 0.01 + 0.2 * on["saturation"] + 0.05 * on["idf"] * on["length"] + 0.03 * on["length"])
    values = shapley(quality)
    assert sum(values.values()) == pytest.approx(quality["111"] - quality["000"])


def test_an_inert_switch_gets_zero():
    quality = lattice(lambda on: 0.1 + 0.3 * on["saturation"] + 0.1 * on["idf"])
    assert shapley(quality)["length"] == pytest.approx(0.0)


def test_additive_effects_are_returned_unchanged():
    quality = lattice(lambda on: 0.2 * on["idf"] + 0.5 * on["saturation"] - 0.1 * on["length"])
    assert shapley(quality) == pytest.approx({"idf": 0.2, "saturation": 0.5, "length": -0.1})


def test_an_interaction_is_split_equally_between_its_switches():
    quality = lattice(lambda on: 0.6 * on["idf"] * on["length"])
    assert shapley(quality) == pytest.approx({"idf": 0.3, "saturation": 0.0, "length": 0.3})


def test_configuration_keys_follow_the_scorer_bit_order():
    assert configuration({"idf"}) == "100" and configuration({"saturation", "length"}) == "011"
