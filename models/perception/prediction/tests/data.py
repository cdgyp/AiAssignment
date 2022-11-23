from ..palette import MaskShiftingUncroppingDataset
import torch
from torch.utils.data import DataLoader

if __name__ == '__main__':
    ds = MaskShiftingUncroppingDataset('../../data/semantic_maps', mask_queue_len=2)
    print(ds.get_ground_truth_identifiers())
    if not ds.is_ground_truth('gt2'):
        gt1 = torch.rand((3, 256, 256))
        ds.add_ground_truth('gt2', gt1)
    for id in ds.get_ground_truth_identifiers():
        mask = torch.rand((256, 256))
        ds.add_mask(id, mask)
        print(f'masks of {id}:', ds._get_masks(id))

        mask = torch.rand((2, 256, 256))
        ds.add_mask(id, mask)
        print(f'masks of {id}:', ds._get_masks(id))