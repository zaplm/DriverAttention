import os.path
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import cv2
from pathlib import Path
import random
from tqdm import tqdm


class StatHard(Dataset):
    def __init__(self, root: str,
                  cam_subdir = 'camera',
                  out_folder = 'infer_gaze',
                  gaze_subdir = 'gaze',
                  mask_subdir = 'masks',
                  sample_list = 'hard_cases_gl4.txt',
                  sample_num = -1,):
        '''
        mode should include:
        train, val, test, infer, run_example
        '''

        self.file_scene_list:list = []
        self.out_folder = out_folder
        self.cam_subdir = cam_subdir
        self.mask_subdir = mask_subdir
        self.gaze_subdir = gaze_subdir
        # self.kl_db:dict = None


        assert os.path.exists(root), f"path '{root}' does not exists."
        self.root = Path(root)
        self.hard_files = None
        scene_list_path = self.root/'test.txt'
        hard_list_path = self.root/sample_list
        
        if hard_list_path.exists():
            with open(hard_list_path) as file:
                self.hard_files = [Path(folder.strip()).stem
                                   for folder in file if not folder.strip().startswith("#")]
                
                
        self.scenes = [self.root/folder.strip()
                    for folder in open(scene_list_path) if not folder.strip().startswith("#")]
        if os.path.exists(str(self.scenes[0]/(self.cam_subdir + '_224_224'))):
                self.cam_subdir = self.cam_subdir + "_224_224"
                print("use resized")
        if os.path.exists(str(self.scenes[0]/(self.gaze_subdir + '_224_224'))):
                self.gaze_subdir = self.gaze_subdir + "_224_224"
                print("use resized")      

        for scene in self.scenes:
            file_list = sorted(
                list((scene/self.cam_subdir).glob('*')))
            file_scene_list = [[x.name, scene.name] for x in file_list if x.suffix in ('.png', '.jpg', '.jpeg')]
            self.file_scene_list = self.file_scene_list + file_scene_list
               
    
    def __getitem__(self, i):
        assert(self.hard_files is not None)
        file = self.hard_files[i]
        scene = file.split('_')[0]
        # file, scene = self.file_scene_list[i]
        ext = '.jpg'
        gaze_path = str(self.root / scene / self.gaze_subdir / file) + ext
        # if self.mode == 'stat':
        #     gaze = self.convert(gaze_path, resize=False)
        #     return gaze
        img_path = str(self.root / scene / self.cam_subdir / file) + ext
        img = self.convert(img_path, True)
        gaze  = self.convert(gaze_path)
        return img, gaze



    def __len__(self):
        assert(self.hard_files is not None)
        return len(self.hard_files)

    @staticmethod
    def convert(x_path, is_rgb=False, resize=True):
        transform_with_resize = transforms.Compose([
            transforms.Resize((224, 224)), # Resize the shorter side to 224, maintaining aspect ratio
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        # transform_with_resize = transforms.Compose([transforms.Resize((224, 224)),
        #                                 transforms.ToTensor(),
        #                                 transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        transform_wo_resize = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        x = Image.open(x_path)
        w, h = x.size
        # print(x.size)

        if is_rgb:
            x = x.convert('RGB')
            if ( w == 224 and h == 224):
                x = transform_wo_resize(x)
            else:
                x = transform_with_resize(x)
        else:
            # print('not rgb')
            x = x.convert('L')
            x = np.array(x)
            x = x.astype('float')
            
            if ( not (w == 224 and h == 224 ) ) and resize==True:
                x = cv2.resize(x, (224, 224))

            if np.max(x) > 1.0:
                x = x / 255.0
            x = torch.FloatTensor(x).unsqueeze(0)

        return x
    
    def calculate_scene_statistics(self):
        scene_statistics = {}
        for scene in tqdm(self.scenes):
            gaze_files = sorted(list((scene / self.gaze_subdir).glob('*.png')) + list((scene / self.gaze_subdir).glob('*.jpg')))
            if not gaze_files:
                print(f"No gaze files found in {scene}")
                continue
            
            # Calculate average gaze value
            avg_gaze = self.calculate_average_gaze(gaze_files)
            
            # Calculate KL divergence for each gaze image and count those > 2
            high_kldiv_count, high_kldiv_files = self.calculate_kldiv(gaze_files, avg_gaze)
            
            scene_statistics[scene.name] = {
                'high_kldiv_count': high_kldiv_count,
                'high_kldiv_files': high_kldiv_files,
            }
            # return scene_statistics
            
        return scene_statistics
    
    def calculate_scene_statistics_global(self):
        scene_statistics = {}
        gaze_files_totl = []
        
        for scene in tqdm(self.scenes):
            gaze_files = sorted(list((scene / self.gaze_subdir).glob('*.png')) + list((scene / self.gaze_subdir).glob('*.jpg')))
            if not gaze_files:
                print(f"No gaze files found in {scene}")
                continue
            gaze_files_totl += gaze_files
        avg_gaze = self.calculate_average_gaze(gaze_files_totl)
        
        
        for scene in tqdm(self.scenes):
            gaze_files = sorted(list((scene / self.gaze_subdir).glob('*.png')) + list((scene / self.gaze_subdir).glob('*.jpg')))
            if not gaze_files:
                print(f"No gaze files found in {scene}")
                continue
            
            # Calculate average gaze value
            # avg_gaze = self.calculate_average_gaze(gaze_files)
            
            # Calculate KL divergence for each gaze image and count those > 2
            high_kldiv_count, high_kldiv_files = self.calculate_kldiv(gaze_files, avg_gaze)
            
            scene_statistics[scene.name] = {
                'high_kldiv_count': high_kldiv_count,
                'high_kldiv_files': high_kldiv_files,
            }
            # return scene_statistics
            
        return scene_statistics
    
    def calculate_average_gaze(self, gaze_files):
        avg_gaze = None
        for gaze_file in gaze_files:
            gaze_image = self.convert(str(gaze_file), is_rgb=False, resize=False)
            if avg_gaze is None:
                avg_gaze = gaze_image
            else:
                avg_gaze += gaze_image
        avg_gaze /= len(gaze_files)
        return avg_gaze
    
    def calculate_kldiv(self, gaze_files, avg_gaze):
        high_kldiv_count = 0
        high_kldiv_files = []
        for gaze_file in gaze_files:
            gaze_image = self.convert(str(gaze_file), is_rgb=False, resize=False)
            kldiv_value = kldiv(gaze_image, avg_gaze)  # Assuming kldiv is a predefined function
            if kldiv_value > 3:
                high_kldiv_count += 1
                high_kldiv_files.append(gaze_file.name)
        return high_kldiv_count, high_kldiv_files

    
def kldiv(s_map, gt):
    batch_size = s_map.size(0)
    # c = s_map.size(1)
    w = s_map.size(1)
    h = s_map.size(2)

    sum_s_map = torch.sum(s_map.view(batch_size, -1), 1)
    expand_s_map = sum_s_map.view(batch_size, 1, 1).expand(batch_size, w, h)

    assert expand_s_map.size() == s_map.size()

    sum_gt = torch.sum(gt.view(batch_size, -1), 1)
    expand_gt = sum_gt.view(batch_size, 1, 1).expand(batch_size, w, h)

    assert expand_gt.size() == gt.size()

    s_map = s_map / (expand_s_map * 1.0)
    gt = gt / (expand_gt * 1.0)

    s_map = s_map.view(batch_size, -1)
    gt = gt.view(batch_size, -1)

    eps = 1e-8
    result = gt * torch.log(eps + gt / (s_map + eps))
    return torch.mean(torch.sum(result, 1))

