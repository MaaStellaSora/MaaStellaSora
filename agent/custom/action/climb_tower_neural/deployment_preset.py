"""Explicit user presets for live inference, separate from learning presets."""
from .critic.presets import Preset, POOL_LENGTHS


class DeploymentPreset(Preset):
    deployment_only = True

    def __post_init__(self):
        pools = tuple(tuple(row) for row in self.pools)
        if tuple(map(len, pools)) != POOL_LENGTHS:
            raise ValueError("实机预设需要全部36个普通目标槽位，包括零目标。")
        if any(type(value) is not int or not 0 <= value <= 6
               for row in pools for value in row):
            raise ValueError("实机目标等级须为0至6的整数。")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("实机预设需要名称。")
        object.__setattr__(self, "pools", pools)
