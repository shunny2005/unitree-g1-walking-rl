"""config.loader.load_config -- focused on pd_gains_left (EXP-0048), the
newest optional field: must default to None (symmetric legs, every
existing config unaffected) and load correctly when present."""

from config.loader import load_config


def test_pd_gains_left_defaults_to_none_for_existing_configs():
    cfg = load_config("config/exp0047_ankle_gain_more_zerocmd.yaml")
    assert cfg.pd_gains_left is None


def test_pd_gains_left_loads_when_present():
    cfg = load_config("config/exp0048_left_ankle_gain.yaml")
    assert cfg.pd_gains_left is not None
    assert cfg.pd_gains_left.ankle_pitch.kp != cfg.pd_gains.ankle_pitch.kp
