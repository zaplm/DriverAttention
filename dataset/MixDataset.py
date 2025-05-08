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
import shutil
from torch.utils.data import DataLoader



class MixDataset(Dataset):
    def __init__(self, root: str, mode: str,
                  cam_subdir = 'camera',
                  gaze_subdir = 'gaze',
                  infer_gaze_subdir = 'infer_gaze',
                  p_dic = ['0', '1'],
                  sample_num = -1,
                  mix_dir='mixup_data'):
        '''
        mode should include:
        train, val, test, infer, run_example
        '''
        self.mode:str = mode

        self.cam_subdir = cam_subdir
        self.gaze_subdir = gaze_subdir
        self.p_dic = p_dic
        self.infer_gaze_subdir = infer_gaze_subdir
        self.file_scene_list = []
        self.out_folder = infer_gaze_subdir
        assert os.path.exists(root), f"path '{root}' does not exists."
        self.root = Path(root)
        self.mix_dir = mix_dir
        self.scenes = [self.root/mix_dir]
        os.makedirs(self.scenes[0], exist_ok=True)
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

        if sample_num != -1:
            #sample sample num
            print(len(self.file_scene_list))
            if sample_num > len(self.file_scene_list ):
                raise ValueError("sample_num is larger than the length of the list")
            self.file_scene_list =  random.sample(self.file_scene_list, sample_num)
        self.init_file_scene_list = self.file_scene_list
        # self.file_scene_list = self.file_scene_list[:1000]

    def clear(self):
        mix_data_dir = (self.root/self.mix_dir) #rm all
        if mix_data_dir.exists() and mix_data_dir.is_dir():
            shutil.rmtree(mix_data_dir)
            print(f"'{mix_data_dir}' directory has been removed.")
        else:
            print(f"'{mix_data_dir}' does not exist or is not a directory.")
    def __getitem__(self, i):
        file, scene = self.file_scene_list[i]
        img_path = str(self.root / scene / self.cam_subdir / file)
        img = self.convert(img_path, True)
        if self.mode == 'train':
            p = []
            for p_type in self.p_dic:
                pesudo_path = str(self.root / scene / p_type / file)
                ps = self.convert(pesudo_path)
                p.append(ps)
            return img, p
        elif self.mode == 'infer':
            output_folder = str(self.root / scene / self.out_folder)
            os.makedirs(output_folder, exist_ok=True)
            out_test_path = str(self.root / scene / self.out_folder / file)
            return img, out_test_path
        elif self.mode == 'cal':
            gaze_path = str(self.root / scene / self.infer_gaze_subdir / file)
            ext = '.jpg'
            gaze_path = str(self.root / scene / self.infer_gaze_subdir / file).rsplit('.', 1)[0] + ext
            gaze  = self.convert(gaze_path)
            return gaze, [file, scene]
        else: 
            raise(NotImplementedError)
        
    

    def get_data_by_kl(self, kl_db:dict, gaze_average)->dict:
        '''
        the function will accept kl_distribution and then try to balance and return the modified kl_db
        only works for aug dataset
        '''
        # import pdb; pdb.set_trace()
        max_value = None
        key_max = None
        
        for key, value in kl_db.items():
            if max_value is None or value > max_value:
                max_value = value
                key_max = key
        
        selected_file_scene_list = []

        mode_saved = self.mode
        batch_size = 32
        num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 16])
        self.mode = 'cal'
        loader = DataLoader(self, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    


        for gazes, [file, scene] in loader:
            gazes = gazes.cuda()
            for idx, gaze in enumerate(gazes):
                kl = kldiv(gaze, gaze_average).item()
                rounded_kl = round(kl, 1)
                if rounded_kl in kl_db  and  kl_db[rounded_kl] < max_value and rounded_kl > key_max:
                    kl_db[rounded_kl] += 1
                    selected_file_scene_list.append([file[idx], scene[idx]])
        self.file_scene_list = selected_file_scene_list

        self.mode = mode_saved
        return kl_db

    def __len__(self):
        # return 100
        # return min(100, len(self.file_scene_list))
        return len(self.file_scene_list)

    @staticmethod
    def convert(x_path, is_rgb=False):
        transform_with_resize = transforms.Compose([
            transforms.Resize((224, 224)), # Resize the shorter side to 224, maintaining aspect ratio
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        transform_wo_resize = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        x = Image.open(x_path)
        w, h = x.size

        if is_rgb:
            x = x.convert('RGB')
            if ( w == 224 and h == 224):
                x = transform_wo_resize(x)
            else:
                x = transform_with_resize(x)
        else:
            x = x.convert('L')
            x = np.array(x)
            x = x.astype('float')
            
            if ( not (w == 224 and h == 224 ) ):
                x = cv2.resize(x, (224, 224))
            if np.max(x) > 1.0:
                x = x / 255.0
            x = torch.FloatTensor(x).unsqueeze(0)
        return x
    
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
    # print(torch.log(eps + gt/(s_map + eps))   )
    return torch.mean(torch.sum(result, 1))
if __name__ == '__main__':
    pass
    
