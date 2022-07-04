from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
import mmcv

import os.path as osp

@DATASETS.register_module()
class ChickenDataset(CustomDataset):
  CLASSES = ("foreground", "background")
  PALETTE = [[255, 255, 255], [0, 0, 0]]
  def __init__(self, split, **kwargs):
    super().__init__(img_suffix='.jpg', seg_map_suffix='.png', 
                     split=split, **kwargs)
    assert osp.exists(self.img_dir) and self.split is not None