from ..palette import prepare
import torch

if __name__ == '__main__':

    model, ds_train, ds_val = prepare(
        class_number=3,
        base_config='models/perception/prediction/palette/config/uncropping_places2.json',
        batch_size=16,
        epoch_per_train=5,
        iter_per_train=5e3,
        gpu_ids=[0],
        phase='train'
    )

    print("initialized")

    model.train()

    image = torch.rand((1, 4, 256, 256)).to('cuda')
    res = model(image)


