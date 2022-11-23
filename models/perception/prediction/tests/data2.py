from ..palette import MaskShiftingUncroppingDataset
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

if __name__ == '__main__':
    ds = MaskShiftingUncroppingDataset('../../data/semantic_maps', mask_queue_len=2)
    print(ds.get_ground_truth_identifiers())
    loader = DataLoader(ds)
    for x in loader:
        print(x['mask'][0].shape)
        plt.imshow(x['mask'][0, 0].detach().numpy())
        plt.show()
        