import torch
from ... import perception
from perception.utils import assert_size, depth_map_to_point_cloud

def local_semantic_map(segmentation: torch.Tensor, depth_map: torch.Tensor, local_map_size, scale, fov, class_number=None) -> torch.Tensor:
    """将视野中的语义分割信息和深度信息转换成局部语义地图
    
    :param torch.Tensor segmentation: BatchSize x H x W
    :param torch.Tensor depth_map: BatchSize x H x W
    :param tuple local_map_size: 输出的局部地图尺寸
    :param float scale: 局部地图中每个像素的大小
    :param tuple fov: (vertical_FOV, horizontal_FOV)
    :param class_number: 类别数量，默认自动从 segmentation 中获取
    :return torch.Tensor: B x ClassNumber x local_map_size[0] x local_map_size[1]
    """
    with torch.no_grad():
        batch_size, H, W = segmentation.shape
        if class_number is None or class_number == 0:
            class_number = int(segmentation.max().item()) + 1
        
        Y, X = depth_map_to_point_cloud(depth_map, fov)
        assert_size(X, (batch_size, H, W))
        assert_size(Y, (batch_size, H, W))
        X, Y = X / scale, Y / scale
        int_X, int_Y = X.type(torch.int), Y.type(torch.int)

        within = torch.logical_and(torch.logical_and(0 <= Y, Y <= local_map_size[0]), torch.logical_and(-local_map_size[0] / 2 <= X, X <= local_map_size[0] / 2))

        batch_id = torch.arange(0, batch_size).reshape(batch_size, 1, 1).expand(-1, H, W)

        within = within.reshape(-1)
        res = torch.zeros((batch_size, class_number, local_map_size[0], local_map_size[1]))
        res[batch_id.reshape(-1)[within], segmentation.reshape(-1)[within], int_Y.reshape(-1)[within], int_X.reshape(-1)[within]] += 1
        return res