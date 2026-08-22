from .data import Data, Potential
from .ui import UIInteractor
from .handler_preset import RecommendationHandler

class RecommendationPlusBagScanHandler(RecommendationHandler):
    def __init__(self, screen: UIInteractor, data: Data):
        super().__init__(screen, data)

    # TODO：背包扫描+推荐潜能选择
    # def _tower_8_score(self, p: Potential) -> int:
    # 因为能获得整体情况，所以还需要考虑下面问题
    # 还得考虑潜能到达旅人上限的问题，以保证推荐潜能数超过普通上限的情况打分偏向于能够突破上限
    # 推荐等级越高分数越高的修正

    # def _tower_8_threshold(self, threshold: int) -> int:
    # 这个可能不用调整思路，但需要把默认16种潜能改为整个潜能库的概率计算
