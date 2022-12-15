import torch
import sys
sys.path.append('../../..')
from models.common import device
import habitat_sim
from matplotlib import pyplot as plt

def assert_size(tensor: torch.Tensor, shape, dim=None):
    shape = torch.Size(shape)
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

def quaternion2radian(agent_rotation):
    cos_half_theta,  sin_half_theta= agent_rotation.w, -agent_rotation.y
    half_theta = torch.tensor(cos_half_theta).acos().abs()
    if sin_half_theta < 0:
        half_theta = -half_theta
    return half_theta * 2

def depth_map_to_point_cloud(depth_map: torch.Tensor, fov: tuple) -> tuple:
    """将深度信息转换成点云

    :param torch.Tensor depth_map: batch_size x H x W
    :param tuple fov: (vertical_FOV, horizontal_FOV) 或 horizontal_FOV
    :return tuple[torch.Tensor, torch.Tensor]: (Y, X), 每个大小都为 batch_size x H x W
    """
    batch_size, H, W = depth_map.shape

    if isinstance(fov, float) or (isinstance(fov, list) and len(fov) <= 1) or (isinstance(fov, torch.Tensor) and (len(fov.shape) == 0 or len(fov) <= 1)):
        # fov 为 hfov
        fov: torch.Tensor = torch.tensor(fov)
        fov = torch.tensor([((fov/2).tan() / depth_map.shape[2] * depth_map.shape[1]).atan() * 2, fov])
    elif not isinstance(fov, torch.Tensor):
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

import torch

def depth_map_to_point_cloud(depth_map: torch.Tensor, fov: tuple) -> tuple:
    """将深度信息转换成点云

    :param torch.Tensor depth_map: batch_size x H x W, 假设深度图中每个像素的值是物体到**成像平面的垂直距离**，而非到焦点的距离
    :param tuple fov: (vertical_FOV, horizontal_FOV) 或 horizontal_FOV，弧度制
    :return tuple[torch.Tensor, torch.Tensor]: (Y, X), 每个大小都为 batch_size x H x W
    """
    with torch.no_grad():
        depth_map = depth_map.to(device).detach()
        batch_size, H, W = depth_map.shape

        if not isinstance(fov, torch.Tensor):
            fov = torch.tensor(fov).to(device)
        tan_half_fov = (fov / 2).tan()

        if len(tan_half_fov.shape) == 0 or len(tan_half_fov) <= 1:
            # tan_half_fov 为 hfov
            tan_half_fov = torch.tensor([tan_half_fov / W * H, tan_half_fov]).to(device)

        xs = torch.arange(0, W).type(torch.float).to(device)
        rxs_wrt_center = (xs - xs.mean()) / ((W - 1) / 2) # 相对坐标；±1 分别对应画布的边界
        assert (rxs_wrt_center.max() - 1).abs() < 1e-5 and (rxs_wrt_center.min() - (-1)).abs() < 1e-5

        xs_wrt_center = (rxs_wrt_center * tan_half_fov[1]).unsqueeze(dim=0).expand(H, -1)
        return depth_map, depth_map * xs_wrt_center.unsqueeze(dim=0)

def fall_on_map(point_cloud: tuple, segmentation: torch.Tensor, map: torch.Tensor, position_of_focus: tuple, scale: float, tmp=None):
    """让点云落在一张地图上

    将来自 `point_cloud` 中的二维点云落在地图 `map` 上。如果点超出 `map` 的范围则会被抛弃

    :param tuple point_cloud: [Y, X]，来自 `depth_map_to_point_cloud` 或 `rotate_point_cloud` 的输出
    :param torch.Tensor segmentation: 语义分割。注意 habitat 的语义传感器给出的结果会为每一个物体（而不是类别）创建一个通道， 不可直接使用，需要使用后面的 `ClassReducer` 将其转换成真正的语义分割
    :param torch.Tensor map: 一张地图，点云会落在该地图上
    :param tuple position_of_focus: [y, x]，相机焦点在地图上的位置，单位：米
    :param float scale: 每个像素的边长的长度，单位：米
    :param torch.Tensor tmp: 一个与 `map` 大小相同的临时全 0 tensor, defaults to None，会由 `fall_on_map` 自己生成、管理
    """
    with torch.no_grad():

        Y, X = point_cloud
        Y: torch.Tensor; X: torch.Tensor 
        batch_size = Y.shape[0]
        
        X, Y = X + position_of_focus[1], position_of_focus[0] - Y
        X, Y = X / scale, Y / scale
        int_X, int_Y = X.type(torch.long), Y.type(torch.long)


        within = (0 <= int_Y) & (int_Y < map.shape[-2]) & (0 <= int_X) & (int_X < map.shape[-1])
        within = within.reshape(batch_size, -1)


        if tmp is None:
            tmp = torch.zeros(map.shape).to(device)
        for t in range(batch_size):
            w = within[t]
            tmp[
                segmentation[t].reshape(-1)[w].type(torch.long), 
                int_Y[t].reshape(-1)[w], 
                int_X[t].reshape(-1)[w]
            ] += 1
            torch.max(map, tmp, out=map)
            tmp[
                segmentation[t].reshape(-1)[w].type(torch.long), 
                int_Y[t].reshape(-1)[w], 
                int_X[t].reshape(-1)[w]
            ] -= 1 # 地图尺寸较大时重复利用比新建更快


def rotate_point_cloud(point_cloud: tuple, rotation:torch.Tensor=torch.tensor(0)) -> tuple:
    """旋转点云

    :param tuple point_cloud: [Y, X]，被旋转的点云。每个坐标的尺寸为 batch_size x H x W。可以直接使用来自 `depth_map_to_point_cloud` 的输出
    :param torch.Tensor rotation: 被旋转的角度，batch 中每个点云可以使用不同的角度, defaults to torch.tensor(0)
    :return tuple: [Y, X]，尺寸同 `point_cloud`
    """
    with torch.no_grad():
        Y, X = point_cloud
        Y: torch.Tensor; X: torch.Tensor 

        batch_size, H, W = Y.shape
        if isinstance(rotation, torch.Tensor):
            if rotation.device != device:
                rotation = rotation.detach().to(device)
        else:
            rotation = torch.tensor(rotation).to(device)
        
        assert_size(X, (batch_size, H, W))
        assert_size(Y, (batch_size, H, W))
        if (rotation != 0).sum() != 0:
            rotation = - rotation
            cos_theta, sin_theta = rotation.cos().reshape(-1, 1, 1), rotation.sin().reshape(-1, 1, 1)
            X, Y = X * cos_theta - Y * sin_theta, X * sin_theta + Y * cos_theta

        return Y, X


class ClassReducer:
    """处理 habitat_sim 使用过程中的类别问题

    使用 habitat_sim 过程中，会遇到两种问题：

    1. 语义传感器给出的分割是物体级别的分割
    2. 潜在的类别过多，但实际出现的类别较少

    使用 ClassReducer 可以

    1. 使用 `.instance_to_category()` 将物体分割粗化成语义分割
    2. 使用 `.reduce()` 将以全部类别标记的语义分割转化成以实际出现类别标记的语义分割
    3. 使用 `.recover()` 和 `.recover_channel()` 将以实际出现类别标记标记的语义分割还原成以全部类别标记的语义分割。其中 `.recover_channel()` 处理以通道方式存储不同类别出现情况的地图
    """
    def __init__(self, matrix: torch.Tensor, class_number=None, instance_to_categories=False) -> None:
        """初始化 ClassReducer

        :param torch.Tensor matrix: 一些真实出现的类别
        :param int class_number: 类别总数, defaults to None：根据 `matrix` 自动获取
        :param bool instance_to_categories: 如果为 `True`，表示可以仰赖 `matrix[i]` 给出第 `i` 个物体的类别编号, defaults to False
        """
        matrix = matrix.type(torch.long).to(device).flatten()
        self.reduced_to_original= matrix.unique(sorted=True, return_inverse=False, return_counts=False)

        if class_number is None:
            class_number = self.reduced_to_original.max().item() + 1
        self.class_number = class_number

        self._instance_to_categories = None
        if instance_to_categories:
            self._instance_to_categories = matrix


        self.original_to_reduced = torch.full((class_number, ), fill_value=-class_number-1).to(device).type(torch.long)
        self.original_to_reduced[self.reduced_to_original.reshape(-1)] = torch.arange(0, len(self.reduced_to_original)).to(device).type(torch.long)
        
        for i in range(len(self.reduced_to_original)):
            assert self.original_to_reduced[self.reduced_to_original[i]] == i, (i, self.reduced_to_original[i].item(), self.original_to_reduced[self.reduced_to_original[i]].item())
        print("reduced number of classes:", len(self.reduced_to_original))
    def from_sim(sim: habitat_sim.Simulator):
        """直接根据模拟器生成 ClassReducer

        :param habitat_sim.Simulator sim: 正在运行的模拟器
        :return ClassReducer:
        """
        categories = torch.tensor([obj.category.index() for obj in sim.semantic_scene.objects]).to(device)
        reducer = ClassReducer(categories, instance_to_categories=True)
        assert (reducer.recover(reducer.reduce(categories)) != categories).sum() == 0
        return reducer
    def get_reduced_class_number(self):
        return len(self.reduced_to_original)
    def save(self, path):
        torch.save(self.reduced_to_original, path)
    def load(path, class_number=None):
        matrix = torch.load(path)
        return ClassReducer(matrix, class_number)
    
    def _map(self, matrix: torch.Tensor, mapping: torch.Tensor):
        if not isinstance(matrix, torch.Tensor):
            matrix = torch.tensor(matrix).to(device)
        matrix = matrix.type(torch.long).to(device)
        assert (matrix < 0).sum() == 0
        assert matrix.max().item() < len(mapping), (matrix.max().item(), len(mapping))
        res = mapping[matrix.reshape(-1)].reshape(matrix.shape)
        assert (res < 0).sum() == 0
        return res
    def instance_to_category(self, instances: torch.Tensor):
        return self._map(instances, self._instance_to_categories)
    def reduce(self, original: torch.Tensor):
        return self._map(original, self.original_to_reduced)
    def recover(self, reduced: torch.Tensor):
        return self._map(reduced, self.reduced_to_original)
    def recover_channel(self, reduced, class_number=None, default_value=0):
        if class_number is None:
            class_number = self.class_number
        res = torch.full(([class_number] + list(reduced.shape[1:])), fill_value=default_value)
        res[self.reduced_to_original] = reduced
        
        return res

def display_semantic_map(topdown: torch.Tensor):
    from habitat_sim.utils.common import d3_40_colors_rgb
    with torch.no_grad():
        remain = topdown.shape[0] % 40
        topdown_main, topdown_remain = topdown[:topdown.shape[0] - remain], topdown[topdown.shape[0] - remain:]
        topdown_main = topdown_main.reshape([topdown.shape[0] // 40, 40] + list(topdown.shape[1:]))
        topdown = topdown_main.max(dim=0)[0]
        topdown[:remain] += topdown_remain
        topdown = topdown.clamp(max=1)

        topdown = (topdown * torch.rand(topdown.shape).to(device)).argmax(axis=0)
        palette = torch.tensor(d3_40_colors_rgb).to(device)
        topdown = palette[(topdown % 40).reshape(-1)].reshape(list(topdown.shape) + [3])

        plt.imshow(topdown.cpu().numpy())
        plt.show()
