from os import environ, path
from typing import Any, Dict, Optional

from dreamerv3 import configs, embodied

from drl_air_hockey.utils.task import Task as AirHockeyTask
from drl_air_hockey.utils.tournament_agent_strategies import (
    strategy_from_str,
    strategy_to_str,
)

ENV = AirHockeyTask.from_str(environ.get("AIR_HOCKEY_ENV", default="tournament"))

AGENT_STRATEGY = strategy_from_str(
    environ.get("AIR_HOCKEY_AGENT_STRATEGY", default="balanced")
)

RENDER: bool = False
INTERPOLATION_ORDER: int = -1

EPISODE_MAX_STEPS: int = 45000
MAX_TIME_UNTIL_PENALTY_S: int = 15.0

DIR_MODELS: str = path.join(
    path.abspath(path.dirname(path.dirname(__file__))),
    "agents",
    "models",
)

def config_dreamerv3(train: bool = False, preset: int = 1, experiment: Optional[str] = None) -> Dict[str, Any]:
    base = dict(configs["defaults"])
    base.update({
    "jax.precision": "float32",
    "jax.platform": "cpu",
    "jax.prealloc": True,
    "imag_horizon": 50,
    "encoder.mlp_keys": "vector",
    "decoder.mlp_keys": "vector",
    "encoder.mlp_layers": 2,
    "encoder.mlp_units": 512,
    "decoder.mlp_layers": 2,
    "decoder.mlp_units": 512,
    "agent.rssm.deter": 512,
    "agent.rssm.units": 512,  # assuming 'units' was meant as hidden
    "agent.rssm.stoch": 32,
    "agent.rssm.classes": 32,
    "agent.dec.mlp_keys": "vector",
    "agent.dec.mlp_layers": 2,
    "agent.dec.mlp_units": 512,
    "agent.enc.mlp_keys": "vector",
    "agent.enc.mlp_layers": 2,
    "agent.enc.mlp_units": 512,
    "agent.disaghead.layers": 2,
    "agent.disaghead.units": 512,
    "actor.layers": 2,
    "actor.units": 512,
    "critic.layers": 2,
    "critic.units": 512,
    "reward_head.layers": 2,
    "reward_head.units": 512,
    "cont_head.layers": 2,
    "cont_head.units": 512,
    "disag_head.layers": 2,
    "disag_head.units": 512,
    "logdir": path.join(
        path.dirname(path.dirname(path.abspath(path.dirname(__file__)))), 
        "logdir_" + ENV.to_str().lower().replace("7dof-", "") + "_" + strategy_to_str(AGENT_STRATEGY),
    )
})
    config = embodied.Config(base)
    ...

    if preset == 1:
        config = config.update({
            "logdir": path.join(
                path.dirname(path.dirname(path.abspath(path.dirname(__file__)))),
                "logdir_" + ENV.to_str().lower().replace("7dof-", "") + "_" + strategy_to_str(AGENT_STRATEGY),
            ),
            "jax": {
                "platform": "cpu",
                "precision": "float32",
                "prealloc": True,
            },
            "imag_horizon": 50,
            "agent": {
                "rssm": {
                    "deter": 512,
                    "units": 512,
                    "stoch": 32,
                    "classes": 32,
                },
                "policy": {
                    "layers": 2,
                    "units": 512
                },
                "value": {
                    "layers": 2,
                    "units": 512
                },
                "rewhead": {
                    "layers": 2,
                    "units": 512
                },
                "conhead": {
                    "layers": 2,
                    "units": 512
                },
                "dec": {
                    "mlp_keys": "vector",
                    "mlp_layers": 2,
                    "mlp_units": 512
                },
                "enc": {
                    "mlp_keys": "vector",
                    "mlp_layers": 2,
                    "mlp_units": 512
                },
                "disaghead": {
                    "layers": 2,
                    "units": 512
                }
            }
        })

    if train:
        num_envs = 10
        #Do the same for this list of configs as you did you did for the portion above
        base = dict(configs["defaults"])
        base.update({
            "jax.platform":"gpu",
            "envs.amount":num_envs,

        })
        config = embodied.Config(base)




        config = config.update({
            "jax": {
                "platform": "gpu"
            },
            "envs": {
                "amount": num_envs
            },
            "run": {
                "actor_batch": num_envs,
                "steps": 5e7,
                "log_every": 1024,
                "train_ratio": 512
            },
            "batch_size": 32,
            "batch_length": 64
        })

    if experiment is not None:
        imag_h = {"imag_horizon_short": 10, "imag_horizon_medium": 25, "imag_horizon_long": 50}.get(experiment)
        if imag_h is not None:
            config = config.update({"imag_horizon": imag_h})
        else:
            raise ValueError(f"Unknown experiment: {experiment}")

    return config
 