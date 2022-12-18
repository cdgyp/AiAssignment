import torch
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))
from models.common import device
import habitat_sim
from matplotlib import pyplot as plt
import json

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


class ClassManager:
    """处理 habitat_sim 使用过程中的类别问题

    使用 habitat_sim 过程中，会遇到三种问题：

    0. 不同场景间的类别编号不一致
    1. 语义传感器给出的分割是物体级别的分割
    2. 全局的潜在的类别过多，但实际出现的类别较少

    使用 ClassManager 可以

    1. 使用 `.instance_to_global()` 将物体分割粗化成语义分割，该语义分割使用在所有场景中一致的类编号
    2. 使用 `.reduce()` 将以全部类别标记的语义分割转化成以实际出现类别标记的语义分割
    3. 使用 `.recover()` 和 `.recover_channel()` 将以实际出现类别标记标记的语义分割还原成以全部类别标记的语义分割。其中 `.recover_channel()` 处理以通道方式存储不同类别出现情况的地图

    使用 `.save()` 和 `.load()`：使用 `.load()` 加载来自 `.save()` 的文件后，无法使用 `.instance_to_global()`
    
    """
    def __init__(self, instance_id: list, instance_class_names: list, instance_class_id: list, path_to_classes_json: str, device=device, instance_availiable=True) -> None:
        """初始化 ClassManager
        """
        with open(path_to_classes_json, 'r') as fp:
            classes_json = json.load(fp)
        assert isinstance(classes_json, dict)

        self.class_number = classes_json['class_number']
        self.instance_availiable = instance_availiable
        class_name_to_class_id = classes_json['class_id']

        instance_number = max(instance_id) + 100
        if isinstance(instance_number, torch.Tensor):
            instance_number = instance_number.item()
        instance_to_global = torch.full([instance_number], class_name_to_class_id['Unknown'])
        for i in range(len(instance_class_id)):
            instance_to_global[instance_id[i]] = class_name_to_class_id[instance_class_names[i]]
        assert (instance_to_global == -1).sum() == 0
        self._instance_to_global = instance_to_global.type(torch.long).to(device)

        local_class_number = max(instance_class_id) + 1
        if isinstance(local_class_number, torch.Tensor):
            local_class_number = local_class_number.item()
        self.local_to_global = torch.full([local_class_number], -1)
        for i in range(len(instance_class_id)):
            assert instance_class_id[i] < local_class_number
            self.local_to_global[instance_class_id[i]] = self._instance_to_global[instance_id[i]]
        assert (self.local_to_global == -1).sum() == 0
        self.local_to_global = self.local_to_global.type(torch.long).to(device)
        
        self.global_to_local = torch.full((self.class_number, ), fill_value=-self.class_number-1).to(device).type(torch.long)
        self.global_to_local[self.local_to_global.reshape(-1)] = torch.arange(0, len(self.local_to_global)).to(device).type(torch.long)
        
        for i in range(len(self.local_to_global)):
            assert self.global_to_local[self.local_to_global[i]] == i, (i, self.local_to_global[i].item(), self.global_to_local[self.local_to_global[i]].item())
        # print("reduced number of classes:", len(self.local_to_global), 'total class number:', self.class_number)

    def from_sim(sim: habitat_sim.Simulator, path_to_classes_json: str, device=device):
        """直接根据模拟器生成 ClassReducer

        :param habitat_sim.Simulator sim: 正在运行的模拟器
        :param str path_to_classes_json: 当前数据集的 `classes.json` 的位置
        :return ClassReducer:
        """
        instance_ids = torch.Tensor([int(obj.id.split("_")[-1]) for obj in sim.semantic_scene.objects]).to(device).type(torch.long)
        class_names = [obj.category.name() for obj in sim.semantic_scene.objects]
        class_ids = torch.Tensor([obj.category.index() for obj in sim.semantic_scene.objects]).to(device).type(torch.long)
        reducer = ClassManager(instance_ids, class_names, class_ids, path_to_classes_json, device=device)
        assert (reducer.reduce(reducer.recover(class_ids)) != class_ids).sum() == 0
        # for i, obj in enumerate(sim.semantic_scene.objects):
            # assert obj.semantic_id < 457
            # assert int(obj.id.split("_")[-1]) < 457
            # assert i < 457
        print(len(class_names), "objects,", reducer.class_number, "classes,", len(reducer.local_to_global), "local classes")
        return reducer
    
    def get_local_class_number(self):
        return len(self.local_to_global)
    def save(self, path):
        torch.save(self.local_to_global, path)

    def load(path: str, path_to_classes_json: str, device=device):
        local_to_global = torch.load(path)
        with open(path_to_classes_json, 'r') as fp:
            class_name_to_id:dict = json.load(fp)['class_id']
        class_id_to_name = {id: name for name, id in class_name_to_id.items()}
        names = [class_id_to_name[id.item()] for id in local_to_global]
        local_ids = torch.arange(0, len(names)).to(device)

        res = ClassManager(local_ids, names, local_ids, path_to_classes_json, device=device, instance_availiable=False)

        assert (res.reduce(res.recover(local_ids)) != local_ids).sum() == 0
        return res
    
    def _map(self, matrix: torch.Tensor, mapping: torch.Tensor):
        if not isinstance(matrix, torch.Tensor):
            matrix = torch.tensor(matrix).to(device)
        matrix = matrix.type(torch.long).to(device)
        assert (matrix < 0).sum() == 0
        assert matrix.max().item() < len(mapping), (len(mapping), [value.item() for value in matrix.flatten() if value.item() >= len(mapping)])
        res = mapping[matrix.reshape(-1)].reshape(matrix.shape)
        return res
    
    def instance_to_global(self, instances: torch.Tensor):
        assert self.instance_availiable
        return self._map(instances, self._instance_to_global)
    def reduce(self, original: torch.Tensor):
        return self._map(original, self.global_to_local)
    def recover(self, reduced: torch.Tensor):
        return self._map(reduced, self.local_to_global)
    def reduce_channel(self, original: torch.Tensor, dim=0):
        assert len(original.shape) > 0
        dim = (len(original.shape) + dim) % len(original.shape)
        if dim == 0:
            return original[self.local_to_global]
        elif dim == 1:
            return original[:, self.local_to_global]
        else:
            raise NotImplementedError
        
    def recover_channel(self, reduced, default_value=0):
        class_number = self.class_number
        res = torch.full(([class_number] + list(reduced.shape[1:])), fill_value=default_value).to(device)
        res[self.local_to_global] = reduced.to(device)
        
        return res

def get_class_number(path: str):
    res = 0
    if os.path.isfile(path):
        if '.reducer' in path:
            reducer = torch.load(path)
            res = max(res, reducer.max().item() + 1)
    else: 
        assert os.path.isdir(path)
        for d in os.listdir(path):
            res = max(get_class_number(os.path.join(path, d)), res)
    return res

def fold_channel(img: torch.Tensor, out_channel: int, reduction='mean'):
    
    remain = img.shape[0] % out_channel
    img_main, img_remain = img[:img.shape[0] - remain], img[img.shape[0] - remain:]
    try:
        fold = img.shape[0] // out_channel
        img_main = img_main.reshape([fold, out_channel] + list(img.shape[1:]))
        if reduction == 'max':
            img = img_main.max(dim=0)[0]
            img[:remain] = torch.max(img[:remain], img_remain)
        elif reduction == 'mean':
            img = img_main.mean(dim=0)
            img[:remain] = (img[:remain] * fold + img_remain) / (fold + 1)

        img = img.clamp(max=1)
    except RuntimeError:
        assert img_main.shape[0] == 0
        img = torch.zeros([out_channel] + list(img.shape[1:]))
        img[:remain] = img_remain
    return img

def display_semantic_map(topdown: torch.Tensor):
    from habitat_sim.utils.common import d3_40_colors_rgb
    with torch.no_grad():
        topdown = fold_channel(topdown, 40, reduction='max')
        topdown = (topdown * torch.rand(topdown.shape).to(device)).argmax(axis=0)
        palette = torch.tensor(d3_40_colors_rgb).to(device)
        topdown = palette[(topdown % 40).reshape(-1)].reshape(list(topdown.shape) + [3])

        plt.imshow(topdown.cpu().numpy())
        plt.show()
