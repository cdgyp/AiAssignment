import os
import sys
sys.path.append('../../')

import tarfile
import habitat_sim
from matplotlib import pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from PIL import Image, ImageOps

def make_cfg(settings: dict, scene_dir: str, scene_name: str):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id =  os.path.join(settings['dataset'], scene_dir, scene_name + '.basis.glb')
    sim_cfg.enable_physics = settings['enable_physics']
    sim_cfg.scene_dataset_config_file = settings['dataset_config']


    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = 'color_sensor'
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [settings['height'], settings['width']]
    color_sensor_spec.position = [0.0, settings['sensor_height'], 0.0]
    color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = 'depth_sensor'
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.position = [0.0, settings['sensor_height'], 0.0]
    depth_sensor_spec.resolution = [settings['height'], settings['width']]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = 'semantic_sensor'
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    sensor_specs = [color_sensor_spec, depth_sensor_spec, semantic_sensor_spec]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=settings['rotation_step'])
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=settings['rotation_step'])
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


# function to display the topdown map

def display_sample(rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([])):
    from habitat_sim.utils.common import d3_40_colors_rgb

    rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

    arr = [rgb_img]
    titles = ["rgb"]
    if semantic_obs.size != 0:
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        arr.append(semantic_img)
        titles.append("semantic")

    if depth_obs.size != 0:
        depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
        arr.append(depth_img)
        titles.append("depth")

    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)
    plt.show(block=False)

def display_nav_mesh(topdown_map, key_points=None):
    # plt.figure(figsize=(12, 8))
    plt.imshow(topdown_map)
    # plot points on map
    if key_points is not None:
        for point in key_points:
            plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)
    plt.show(block=False)

class fulfill:
    def __init__(self, dataset_dir: str, scene_dir: str) -> None:
        self.dataset = dataset_dir
        self.scene_dir = scene_dir
        self.expected_path = os.path.join(self.dataset, self.scene_dir)
        self.files_to_delete = []
    def __enter__(self):
        basis = [c for c in os.listdir(self.expected_path) if 'basis' in c]
        
        in_tar = len(basis) <= 0
        if in_tar:
            with tarfile.open(os.path.join(self.dataset, 'main.tar')) as tar:
                for mem in [file for file in tar.getnames() if self.scene_dir + '/' in file]:
                    tar.extract(mem, self.dataset)
                    if not os.path.isdir(os.path.join(self.dataset, mem)):
                        self.files_to_delete.append(mem)
            assert len(self.files_to_delete) > 0

        
    def __exit__(self, type, value, backtrace):
        for file in self.files_to_delete:
            print("deleting", file)
            os.remove(os.path.join(self.dataset, file))
        self.files_to_delete = []

def _for_each_dir(for_each_dir, settings, scene_dir, scene_name=None):
    if scene_name is None:
        scene_name = scene_dir[scene_dir.find('-') + 1:]
    cfg = make_cfg(settings, scene_dir, scene_name)
    sim = habitat_sim.Simulator(cfg)

    for_each_dir(sim, scene_dir, scene_name)

    sim.close()

def make_samples(settings: dict, for_each_dir=lambda sim, scene_dir, scene_name: None):
    # 无论 main.tar 是否解压，semantic.tar 都会给出有语义标记的目录
    from ...models.common import press_habitat_log
    press_habitat_log()
    dirs = []
    for dir in os.listdir(settings['dataset']):
        if not os.path.isdir(os.path.join(settings['dataset'], dir)):
            continue

        contents = os.listdir(os.path.join(settings['dataset'], dir))
        found_semantic = False
        for c in contents:
            if 'semantic' in c:
                found_semantic = True
        if not found_semantic:
            continue
        dirs.append(dir)
    resume = 0
    if 'resume' in settings:
        resume = settings['resume']
    for i, dir in enumerate(tqdm(dirs)):
        print(dir)
        if i < resume:
            print("done")
            continue
        with fulfill(settings['dataset'], dir):
            _for_each_dir(for_each_dir, settings, dir)