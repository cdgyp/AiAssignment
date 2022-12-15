import os
import tarfile
import tempfile

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


