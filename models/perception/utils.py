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

def depth_map_to_point_cloud(depth_map: torch.Tensor, fov: tuple) -> tuple[torch.Tensor, torch.Tensor]:
    """将深度信息转换成点云

    :param torch.Tensor depth_map: batch_size x H x W
    :param tuple fov: (vertical_FOV, horizontal_FOV)
    :return tuple[torch.Tensor, torch.Tensor]: (Y, X), 每个大小都为 batch_size x H x W
    """
    batch_size, H, W = depth_map.shape

    if not isinstance(fov, torch.Tensor):
        fov = torch.tensor(fov)
    tan_half_fov = (fov / 2).tan()
    ys, xs = torch.arange(0, H), torch.arange(0, W)
    rys_wrt_center, rxs_wrt_center = (ys - ys.mean()) / (H / 2), (xs - xs.mean()) / (W / 2) # 相对坐标；±1 分别对应画布的边界
    assert (rys_wrt_center.max() - 1).abs() < 1e-5 and (rys_wrt_center.min() - (-1)).abs() < 1e-5
    assert (rxs_wrt_center.max() - 1).abs() < 1e-5 and (rxs_wrt_center.min() - (-1)).abs() < 1e-5

    ys_wrt_center, xs_wrt_center = (rys_wrt_center * tan_half_fov[0]).unsqueeze(dim=1).expand(-1, W), (rxs_wrt_center * tan_half_fov[1]).unsqueeze(dim=0).expand(H, -1)
    dist_to_focus = (ys_wrt_center ** 2 + xs_wrt_center ** 2 + 1).sqrt() # 画布到焦点的距离为 1
    assert_size(dist_to_focus, depth_map.shape[1:])

    ratio = (depth_map / dist_to_focus.unsqueeze(dim=0)) 
    return ratio * ys_wrt_center.unsqueeze(dim=0), ratio * xs_wrt_center.unsqueeze(dim=0)
