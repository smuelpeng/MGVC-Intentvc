import os
from dataclasses import dataclass, field
from datetime import datetime

from omegaconf import OmegaConf, DictConfig
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    NamedTuple,
    NewType,
    Optional,
    Sized,
    Tuple,
    Type,
    TypeVar,
    Union,
)

# ============ Register OmegaConf Recolvers ============= #
OmegaConf.register_new_resolver(
    "calc_exp_lr_decay_rate", lambda factor, n: factor ** (1.0 / n)
)
OmegaConf.register_new_resolver("add", lambda a, b: a + b)
OmegaConf.register_new_resolver("sub", lambda a, b: a - b)
OmegaConf.register_new_resolver("mul", lambda a, b: a * b)
OmegaConf.register_new_resolver("div", lambda a, b: a / b)
OmegaConf.register_new_resolver("idiv", lambda a, b: a // b)
OmegaConf.register_new_resolver("basename", lambda p: os.path.basename(p))
OmegaConf.register_new_resolver("rmspace", lambda s, sub: s.replace(" ", sub))
OmegaConf.register_new_resolver("tuple2", lambda s: [float(s), float(s)])
OmegaConf.register_new_resolver("gt0", lambda s: s > 0)
OmegaConf.register_new_resolver("cmaxgt0", lambda s: C_max(s) > 0)
OmegaConf.register_new_resolver("not", lambda s: not s)
OmegaConf.register_new_resolver(
    "cmaxgt0orcmaxgt0", lambda a, b: C_max(a) > 0 or C_max(b) > 0
)
OmegaConf.register_new_resolver("ftrest", lambda sh_degree: (sh_degree + 1) ** 2 * 3 - 3)
OmegaConf.register_new_resolver("eq", lambda x, y: x == y)
OmegaConf.register_new_resolver("patch", lambda x: int(x[-2:]))
# ======================================================= #


def C_max(value: Any) -> float:
    if isinstance(value, int) or isinstance(value, float):
        pass
    else:
        value = config_to_primitive(value)
        if not isinstance(value, list):
            raise TypeError("Scalar specification only supports list, got", type(value))
        if len(value) == 3:
            value = [0] + value
        assert len(value) == 4
        start_step, start_value, end_value, end_step = value
        value = max(start_value, end_value)
    return value


@dataclass
class ExperimentConfig:
    name: str = "default"

    resume: Optional[str] = None

    selector_type: str = ""
    selector: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.tag and not self.use_timestamp:
            raise ValueError("Either tag is specified or use_timestamp is True.")
        self.trial_name = self.tag
        # if resume from an existing config, self.timestamp should not be None
        if self.timestamp is None:
            self.timestamp = ""
            if self.use_timestamp:
                if self.n_gpus > 1:
                    print(
                        "Timestamp is disabled when using multiple GPUs, please make sure you have a unique tag."
                    )
                else:
                    self.timestamp = datetime.now().strftime("@%Y%m%d-%H%M%S")
        self.trial_name += self.timestamp
        self.exp_dir = os.path.join(self.exp_root_dir, self.name)
        self.trial_dir = os.path.join(self.exp_dir, self.trial_name)
        os.makedirs(self.trial_dir, exist_ok=True)


def load_config(*yamls: str, cli_args: list = [], from_string=False, **kwargs) -> Any:
    if from_string:
        yaml_confs = [OmegaConf.create(s) for s in yamls]
    else:
        yaml_confs = [OmegaConf.load(f) for f in yamls]
    cli_conf = OmegaConf.from_cli(cli_args)
    cfg = OmegaConf.merge(*yaml_confs, cli_conf, kwargs)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    scfg = parse_structured(ExperimentConfig, cfg)
    return scfg


def config_to_primitive(config, resolve: bool = True) -> Any:
    return OmegaConf.to_container(config, resolve=resolve)


def dump_config(path: str, config) -> None:
    with open(path, "w") as fp:
        OmegaConf.save(config=config, f=fp)


def parse_structured(fields: Any, cfg: Optional[Union[dict, DictConfig]] = None) -> Any:
    scfg = OmegaConf.structured(fields(**cfg))
    return scfg
