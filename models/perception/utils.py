import torch
def assert_size(tensor: torch.Tensor, shape, dim=None):
    shape = list(shape)
    flag = True
    if dim is None:
        flag = (tensor.shape == shape)
    else:
        try:
            for d in dim:
                if tensor.shape[d] != shape[d]:
                    flag = False
        except IndexError:
            flag = False
    
    assert flag, f"expected shape: {shape}, actual shape: {tensor.shape}"


def depth_map_to_point_cloud(depth_map: torch.Tensor, ) -> tuple[torch.Tensor, torch.Tensor]:
    """将深度信息转换成点云

    :param torch.Tensor depth_map: batch_size x H x W
    :return tuple[torch.Tensor, torch.Tensor]: (X, Y), 每个大小都为 batch_size x H x W
    """
    ...

