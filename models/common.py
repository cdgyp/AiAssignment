import torch
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
goal_categories = ["chair", "couch", "bed", "toilet", "tv"]

def press_habitat_log():
    os.environ["MAGNUM_LOG"] = "quiet"
    os.environ["HABITAT_SIM_LOG"] = "quiet"