"""Тести вибору/конфігурації політики RL (ml/rl/policies.py)."""

from omegaconf import OmegaConf

from dronesim.ml.rl.policies import policy_kwargs_from_config, select_policy_name


def test_select_policy_name_defaults_to_mlp():
    assert select_policy_name() == "MlpPolicy"
    assert select_policy_name(observation_is_image=False) == "MlpPolicy"


def test_select_policy_name_cnn_for_image_obs():
    assert select_policy_name(observation_is_image=True) == "CnnPolicy"


def test_policy_kwargs_reads_net_arch_from_config():
    cfg = OmegaConf.create({"net_arch": [128, 128]})
    kwargs = policy_kwargs_from_config(cfg)
    assert kwargs == {"net_arch": [128, 128]}
