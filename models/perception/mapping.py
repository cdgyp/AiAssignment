import torch
from utils import assert_size, depth_map_to_point_cloud

def local_semantic_map(segmentation: torch.Tensor, depth_map: torch.Tensor, local_map_size, scale, class_number=None) -> torch.Tensor:
    """将视野中的语义分割信息和深度信息转换成局部语义地图
    
    :param torch.Tensor segmentation: BatchSize x H x W
    :param torch.Tensor depth_map: BatchSize x H x W
    :param tuple local_map_size: 输出的局部地图尺寸
    :param float scale: 局部地图中每个像素的大小
    :param class_number: 类别数量，默认自动从 segmentation 中获取
    :return torch.Tensor: B x local_map_size[0] x local_map_size[1] x ClassNumber
    """
    ...