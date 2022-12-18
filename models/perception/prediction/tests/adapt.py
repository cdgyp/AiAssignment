if __name__ == '__main__':
    import torch
    torch.multiprocessing.set_start_method('spawn')
    from ..palette import prepare

    model, ds = prepare(
        class_number=48,
        base_config='models/perception/prediction/palette/config/uncropping_places2.json',
        batch_size=2,
        epoch_per_train=1e3,
        iter_per_train=1e8,
        gpu_ids=[0],
        phase='train'
    )

    print("initialized")
    print(ds.get_ground_truth_identifiers())

    model.train()

    # image = torch.rand((1, 1409, 256, 256)).to('cuda')
    # res = model(image)


