import os.path
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import cv2
from pathlib import Path
import random
from torch.utils.data import DataLoader


class SceneDatasetCor(Dataset):
    def __init__(self, root: str, mode: str,
                  alpha: float = 0.3,
                  severity:str = None,
                  cam_subdir = 'camera',
                  out_folder = 'infer_gaze',
                  infer_gaze_subdir = 'infer_gaze',
                  gaze_subdir = 'gaze',
                  mask_subdir = 'masks',
                  p_dic = None,
                  sample_num = -1,
                  noise_type:str = None):
        '''
        mode should include:
        train, val, test, infer, run_example
        '''
        self.mode:str = mode

        self.file_scene_list:list = []
        self.out_folder = out_folder
        self.cam_subdir = cam_subdir
        self.mask_subdir = mask_subdir
        self.gaze_subdir = gaze_subdir
        self.severity = severity
        self.noise_type = noise_type
        self.p_dic:list = p_dic
        self.use_msk = True
        self.alpha =alpha
        self.infer_gaze_subdir = infer_gaze_subdir
        self.gaze_average = None
        # self.kl_db:dict = None


        assert os.path.exists(root), f"path '{root}' does not exists."
        self.root = Path(root)
        phase = mode
        if mode.startswith('infer'):
            phase = 'train'
        scene_list_path = self.root/(phase + '.txt') if self.noise_type is None else  self.root/(phase + '_cor.txt')
        self.scenes = [self.root/folder.strip()
                    for folder in open(scene_list_path) if not folder.strip().startswith("#")]
        if os.path.exists(str(self.scenes[0]/(self.cam_subdir + '_224_224'))):
                self.cam_subdir = self.cam_subdir + "_224_224"
                print("use resized")
        if os.path.exists(str(self.scenes[0]/(self.gaze_subdir + '_224_224'))):
                self.gaze_subdir = self.gaze_subdir + "_224_224"
                print("use resized")      
        if self.p_dic is not None:
            for p in self.p_dic:
                if os.path.exists(str(self.scenes[0]/(p + '_224_224'))):
                        p += "_224_224"
                        print("use resized")  

        if self.noise_type is  None:
            for scene in self.scenes:
                file_list = sorted(
                    list((scene/self.cam_subdir).glob('*')))
                file_scene_list = [[x.name, scene.name] for x in file_list if x.suffix in ('.png', '.jpg', '.jpeg')]
                self.file_scene_list = self.file_scene_list + file_scene_list
        else:
            if not os.path.exists(str(self.scenes[0]/(self.noise_type))):
                self.noise_type = self.noise_type + "_224_224"
            self.out_folder += ('_' + self.noise_type)

            print("Noise Loader")
            for scene in self.scenes:
                file_list = sorted(
                    list((scene/self.noise_type).glob('*')))
                file_scene_list = [[x.name, scene.name] for x in file_list if x.suffix in ('.png', '.jpg', '.jpeg')]
                self.file_scene_list = self.file_scene_list + file_scene_list

        if sample_num != -1:
            print(len(self.file_scene_list))
            if sample_num > len(self.file_scene_list ):
                raise ValueError("sample_num is larger than the length of the list")
            self.file_scene_list =  random.sample(self.file_scene_list, sample_num)
        # self.file_scene_list = self.file_scene_list[:10] #DEBUG
        self.init_file_scene_list = self.file_scene_list
        # self.file_scene_list = self.file_scene_list[:1000]

    def change_mode(self, mode):
        self.mode = mode
    def __getitem__(self, i):
        file, scene = self.file_scene_list[i]
        if self.noise_type:
            img_path = str(self.root / scene / self.noise_type / file)
        else:
            img_path = str(self.root / scene / self.cam_subdir / file)
        img = self.convert(img_path, True)
        if self.mode == 'train':
            p = []
            for p_type in self.p_dic:
                pesudo_path = str(self.root / scene / p_type / file)
                ps = self.convert(pesudo_path)
                if self.use_msk:
                    mask_path = str(self.root / scene / self.mask_subdir / file)
                    mall = self.convert(mask_path)
                    ps = ps * (mall + self.alpha)
                    ps /= ps.max()
                p.append(ps)

            return img, p
        elif self.mode.startswith('val') or self.mode == 'test':
            gaze_path = str(self.root / scene / self.gaze_subdir / file)
            extensions = ['.png', '.jpg', '.jpeg']

            # Iterate through the extensions and check if the file exists
            for ext in extensions:
                gaze_path = str(self.root / scene / self.gaze_subdir / file).rsplit('.', 1)[0] + ext
                if os.path.exists(gaze_path):
                    break
            gaze  = self.convert(gaze_path)
            return img, gaze
        
        elif self.mode == 'run_example' or self.mode =='infer' or self.mode=='test_nobias':
            output_folder = str(self.root / scene / self.out_folder)
            os.makedirs(output_folder, exist_ok=True)
            out_test_path = str(self.root / scene / self.out_folder / file)
            return img, out_test_path
        elif self.mode == 'vis':
            output_folder = str(self.root / scene / self.out_folder)
            os.makedirs(output_folder, exist_ok=True)
            out_test_path = str(self.root / scene / self.out_folder / file)
            return img, out_test_path
        elif self.mode == 'infer_mix':
            output_folder = str(self.root / scene / self.out_folder)
            os.makedirs(output_folder, exist_ok=True)
            out_test_path = str(self.root / scene / self.out_folder / file)

            p = []
            for p_type in self.p_dic:
                pesudo_path = str(self.root / scene / p_type / file)
                ps = self.convert(pesudo_path)
                if self.use_msk:
                    mask_path = str(self.root / scene / self.mask_subdir / file)
                    mall = self.convert(mask_path)
                    ps = ps * (mall + self.alpha)
                    ps /= ps.max()
                p.append(ps)
            return img, p, out_test_path
        elif self.mode == 'cal':
            gaze_path = str(self.root / scene / self.infer_gaze_subdir / file)
            ext = '.jpg'
            gaze_path = str(self.root / scene / self.infer_gaze_subdir / file).rsplit('.', 1)[0] + ext
            gaze  = self.convert(gaze_path)
            return gaze

        else: 
            raise(NotImplementedError)
        

    def set_gaze_average(self, gaze_average):
        self.gaze_average = gaze_average

    def cal_data_kl(self, name)->dict:
        '''
        only used for the init datasets#TODO modify it to cuda
        '''
        mode_saved = self.mode
        batch_size = 32
        num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 16])
        self.mode = 'cal'
        loader = DataLoader(self, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        if self.gaze_average is None:
            sum_gaze = 0
            total_samples = 0
            # gaze_average = None
            for gazes in loader:
                gazes = gazes.cuda()
                batch_sum = gazes.sum(dim=0) 
                sum_gaze += batch_sum
                total_samples += gazes.size(0)  
                    
            self.gaze_average = sum_gaze / total_samples
            out = self.gaze_average.permute(1, 2, 0).cpu().detach().numpy() * 255
            cv2.imwrite(name + '_gaze_average.jpg', out)

        kl_db = {}

        for gazes in loader:
            gazes = gazes.cuda()
            for gaze in gazes:
                kl = kldiv(gaze, self.gaze_average).item()
                rounded_kl = round(kl, 1)
                if rounded_kl in kl_db:
                    kl_db[rounded_kl] += 1
                else:
                    kl_db[rounded_kl] = 1

        self.mode = mode_saved
        return kl_db, self.gaze_average
    


        # for  file, scene in self.init_file_scene_list:

        #     ext = '.jpg'
        #     gaze_path = str(self.root / scene / self.infer_gaze_subdir / file).rsplit('.', 1)[0] + ext
        #     gaze  = self.convert(gaze_path).cuda()
        #     kl = kldiv(gaze_average, gaze).item()
        #     rounded_kl = round(kl, 1)
        #     if rounded_kl in kl_db  and  kl_db[rounded_kl] < max_value and rounded_kl > key_max:
        #         kl_db[rounded_kl] += 1
        #         selected_file_scene_list.append([file, scene])
        # self.file_scene_list = selected_file_scene_list
        # return kl_db



    def __len__(self):
        return len(self.file_scene_list)

    @staticmethod
    def convert(x_path, is_rgb=False, is_msk=False):
        transform_with_resize = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
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
            if is_msk:
                # print('true')
                # x[x == 6] = 255
                roi = np.array([6, 7, 11, 17, 18])
                x = np.isin(x, roi)
                # x = x.astype('int')
                # print(x)
            x = x.astype('float')
            
            if ( w != 224 or h != 224):
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
    return torch.mean(torch.sum(result, 1))
