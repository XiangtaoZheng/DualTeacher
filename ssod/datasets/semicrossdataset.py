from mmdet.datasets import DATASETS, ConcatDataset, build_dataset


@DATASETS.register_module()
class SemiCrossDataset(ConcatDataset):
    """Wrapper for semisupervised od."""

    def __init__(self, sup1: dict, sup2: dict, unsup: dict, **kwargs):
        super().__init__([build_dataset(sup1), build_dataset(sup2), build_dataset(unsup)], **kwargs)

    @property
    def sup1(self):
        return self.datasets[0]

    @property
    def sup2(self):
        return self.datasets[1]

    @property
    def unsup(self):
        return self.datasets[2]

