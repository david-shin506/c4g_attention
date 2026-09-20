import time
from dataclasses import fields

from torch.utils.data import Dataset

from ..misc.step_tracker import StepTracker
from .dataset_adt import DatasetADT, DatasetADTCfg, DatasetADTCfgWrapper
from .dataset_nvidia import DatasetNvidia, DatasetNvidiaCfg, DatasetNvidiaCfgWrapper
from .dataset_tum import DatasetTUM, DatasetTUMCfg, DatasetTUMCfgWrapper
from .dataset_iphone import DatasetIphone, DatasetIphoneCfg, DatasetIphoneCfgWrapper
from .dataset_re10k import DatasetRE10k, DatasetRE10kCfg, DatasetRE10kCfgWrapper
from .dataset_spring import DatasetSpring, DatasetSpringCfg, DatasetSpringCfgWrapper
from .dataset_kubric import DatasetKubric, DatasetKubricCfg, DatasetKubricCfgWrapper
from .dataset_egoexo4d import DatasetEgoExo4D, DatasetEgoExo4DCfg, DatasetEgoExo4DCfgWrapper
from .dataset_egoexo4d_mono import DatasetEgoExo4DMono, DatasetEgoExo4DMonoCfg, DatasetEgoExo4DMonoCfgWrapper
from .types import Stage
from .view_sampler import get_view_sampler

DATASETS: dict[str, Dataset] = {
    "adt": DatasetADT,
    "nvidia": DatasetNvidia,
    "tum": DatasetTUM,
    "iphone": DatasetIphone,
    "re10k": DatasetRE10k,
    "spring": DatasetSpring,
    "kubric": DatasetKubric,
    "egoexo4d": DatasetEgoExo4D,
    "egoexo4d_mono": DatasetEgoExo4DMono,
}


DatasetCfgWrapper = (
    DatasetADTCfgWrapper
    | DatasetNvidiaCfgWrapper
    | DatasetTUMCfgWrapper
    | DatasetIphoneCfgWrapper
    | DatasetSpringCfgWrapper
    | DatasetKubricCfgWrapper
    | DatasetRE10kCfgWrapper
    | DatasetEgoExo4DCfgWrapper
    | DatasetEgoExo4DMonoCfgWrapper
)

DatasetCfg = (
    DatasetADTCfg
    | DatasetNvidiaCfg
    | DatasetTUMCfg
    | DatasetIphoneCfg
    | DatasetRE10kCfg
    | DatasetSpringCfg
    | DatasetKubricCfg
    | DatasetEgoExo4DCfg
    | DatasetEgoExo4DMonoCfg
)


def get_dataset(
    cfgs: list[DatasetCfgWrapper],
    stage: Stage,
    step_tracker: StepTracker | None,
) -> list[Dataset]:
    datasets = []
    for cfg in cfgs:
        (field,) = fields(type(cfg))
        cfg = getattr(cfg, field.name)

        view_sampler = get_view_sampler(
            cfg.view_sampler,
            stage,
            cfg.overfit_to_scene is not None,
            cfg.cameras_are_circular,
            step_tracker,
        )
        print(f"{cfg.name} dataset initializing with view sampler {cfg.view_sampler}...")
        t_start = time.time()
        dataset = DATASETS[cfg.name](cfg, stage, view_sampler)
        datasets.append(dataset)
        print(f"{cfg.name} dataset initialized in {time.time() - t_start:.1f}s")

    return datasets
