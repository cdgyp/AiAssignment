import os
import torch

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES']='2'
    from ..palette import prepare

    torch.cuda.set_device(0)
    model, ds = prepare(
        class_number=17,
        base_config='models/perception/prediction/palette/config/simple_uncropping.json',
        batch_size=4,
        epoch_per_train=1e4,
        iter_per_train=1e9,
        gpu_ids=[2],
        phase='train'
    )

    print("initialized")
    print(ds.get_ground_truth_identifiers())

    model.train()

    # image = torch.rand((1, 1409, 256, 256)).to('cuda')
    # res = model(image)


