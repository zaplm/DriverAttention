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
from tqdm import tqdm


class SceneDataset(Dataset):
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
                  noise_type:str = None,
                  use_prior=True):
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
        self.infer_gaze_subdir = infer_gaze_subdir
        self.gaze_average = None
        self.use_prior = use_prior


        assert os.path.exists(root), f"path '{root}' does not exists."
        self.root = Path(root)
        phase = mode
        if mode.startswith('infer'):
            phase = 'train'
        scene_list_path = self.root/(phase + '.txt') if self.noise_type is None else  self.root/(phase + '_cor.txt')
        # scene_list_path = self.root/('train' + '.txt') 
        
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
    
    def get_weighted_prior(self, scene, file):
        if 'bd' in self.root.name:
            ratio_dic = {'car': 0.0, 'person': 0.18122506607338648, 'stop sign': 0.19527073080923388, 'traffic light': 0.2616087945898801, 'truck': 0.17388554465132544, 'bicycle': 0.16350374183093028, 'bus': 0.16153357425939865, 'motorcycle': 0.19183225919958863, 'bench': 0.10643748839719337, 'dog': 0.05576605004173327, 'backpack': 0.09777960469388935, 'fire hydrant': 0.06474239858484047, 'handbag': 0.08763731467521027, 'parking meter': 0.16562452863722288, 'potted plant': 0.06092319412051039, 'cup': 0.11359222911416274, 'skateboard': 0.09840303970349075, 'train': 0.20691480532578713, 'tv': 0.10791136824738431, 'clock': 0.1144762219590874, 'cell phone': 0.13222140138030689, 'umbrella': 0.11024437257081295, 'bird': 0.08634613909001014, 'boat': 0.18348954395743725, 'kite': 0.03687248143775576, 'suitcase': 0.09635855838609045, 'bowl': 0.0666870674151306, 'chair': 0.05954423152044212, 'frisbee': 0.084945172864205, 'cow': 0.06745842585058782, 'airplane': 0.12468087536177429, 'horse': 0.03908503110024358, 'sink': 0.06261205817910573, 'surfboard': 0.10702988498317394, 'keyboard': 0.0785430556265317, 'mouse': 0.024297037773953804, 'sports ball': 0.14885588348762518, 'elephant': 0.05967972605245516, 'tennis racket': 0.0650258207044329, 'banana': 0.10106796738261853, 'baseball bat': 0.017516029519683153, 'bottle': 0.07283924099191112, 'broccoli': 0.0, 'book': 0.03529503115137245, 'toilet': 0.07208270054966644, 'fork': 0.11649917077285467, 'vase': 0.04720483379038387, 'knife': 0.0703967826049966, 'refrigerator': 0.026514250192185052, 'cat': 0.0, 'microwave': 0.07097810306040395, 'tie': 0.0603520078010561, 'oven': 0.008240488923692621, 'snowboard': 0.11479120579603207, 'bed': 0.03127203895436038, 'dining table': 0.02025312469653382, 'sheep': 0.03619519410151036}
        elif 'da' in self.root.name:
            ratio_dic = {'car': 0.0, 'truck': 0.12233990644872318, 'boat': 0.11889032732936781, 'bird': 0.07851308977918361, 'bus': 0.12652035628231068, 'baseball bat': 0.016092012642318586, 'suitcase': 0.06380411248105904, 'bench': 0.07213681909939261, 'traffic light': 0.2526006226737677, 'motorcycle': 0.04000062605945449, 'bicycle': 0.0999279690044728, 'handbag': 0.03673357257917106, 'stop sign': 0.19137257129395455, 'clock': 0.08126338507859024, 'sports ball': 0.09351953160756028, 'train': 0.15864641163945345, 'cow': 0.02266318892590401, 'dog': 0.038952141930354484, 'fire hydrant': 0.04784994048614122, 'tennis racket': 0.03715364201034535, 'kite': 0.03461525600052598, 'umbrella': 0.07399793966812152, 'potted plant': 0.04613022519507252, 'frisbee': 0.0474648686550058, 'elephant': 0.026415546098585978, 'parking meter': 0.11700506582471897, 'airplane': 0.08274464747236501, 'tv': 0.07760018455949987, 'wine glass': 0.008617173546259575, 'chair': 0.046918326305137134, 'cat': 0.0, 'refrigerator': 0.0237861717560917, 'horse': 0.004946206395584523, 'backpack': 0.05247194822718696, 'cake': 0.004415864331825077, 'cup': 0.09077935453810192, 'vase': 0.03926609177850628, 'bottle': 0.059923102527903595, 'bowl': 0.034119418991227154, 'skateboard': 0.040952649679320274, 'keyboard': 0.06971299378726462, 'sink': 0.05239895547311636, 'pizza': 0.0, 'toilet': 0.06129980825682182, 'oven': 0.007289961668588616, 'giraffe': 0.007286591769715921, 'cell phone': 0.08368164524623845, 'banana': 0.061566586655638356, 'bed': 0.02563476054629472, 'tie': 0.05681178295800592, 'teddy bear': 0.0, 'book': 0.027242075237345655, 'snowboard': 0.12239039153559002, 'microwave': 0.06506587853234529, 'dining table': 0.01716254814308018, 'person': 0.07240239209321829}
        elif 'dr' in self.root.name:
            ratio_dic =  {'car': 0.0, 'clock': 0.13874513031120167, 'boat': 0.20652611439770596, 'traffic light': 0.2793526645914734, 'bird': 0.10541088235025943, 'kite': 0.045358775949868285, 'stop sign': 0.22347788085993686, 'chair': 0.07118423588463459, 'potted plant': 0.06802263254782581, 'truck': 0.15994634319161627, 'bench': 0.11375420618380495, 'cow': 0.07249573723082534, 'airplane': 0.1211525517500804, 'fire hydrant': 0.05416122215627682, 'bicycle': 0.18245995512470684, 'elephant': 0.0461137727741646, 'motorcycle': 0.2096034153552822, 'sheep': 0.03565109520714902, 'bus': 0.16418370685195002, 'train': 0.2313256216859091, 'frisbee': 0.07673088929327292, 'parking meter': 0.1864185431551573, 'sports ball': 0.14194220511623945, 'bottle': 0.04766875180409624, 'umbrella': 0.13314599797437493, 'cup': 0.10515788980993983, 'tv': 0.11796118643341778, 'sink': 0.06192111547981831, 'handbag': 0.06985666858240488, 'suitcase': 0.09288495126253844, 'backpack': 0.09818299954661222, 'skateboard': 0.12107334614490044, 'bowl': 0.07436673974567512, 'baseball bat': 0.022066766514106866, 'mouse': 0.02483641627480934, 'spoon': 0.07930684138540264, 'horse': 0.06003471531785409, 'teddy bear': 0.0, 'baseball glove': 0.05209084968095902, 'tennis racket': 0.08052118189120355, 'oven': 0.009157972665070149, 'banana': 0.12876066123707858, 'knife': 0.08658758337612164, 'vase': 0.044078110698021034, 'laptop': 0.018785659422779277, 'wine glass': 0.10568526398700703, 'dog': 0.07832242288218738, 'surfboard': 0.1282058543001177, 'toilet': 0.08408924168640101, 'refrigerator': 0.03284695736976345, 'bed': 0.038786309394509284, 'book': 0.04132156427101365, 'fork': 0.13774055302277896, 'person': 0.1954076988613516}

        mall = torch.zeros(1, 224, 224)
        masks_dir=self.root / scene / 'masks_stat' / file
        masks_path = sorted(
                    list((masks_dir).glob('*')))
        
        for msk_p in masks_path:
            category = msk_p.stem.split('_')[0]  
            msk = self.convert(msk_p)
            index = ratio_dic[category] if category in ratio_dic else 0
            mall = mall + msk * index
        return mall
    
    def get_pseudos(self, scene, file):
        p = []
        if self.use_prior:
            mall = self.get_weighted_prior(scene, file)
        
        for p_type in self.p_dic:
            pesudo_path = str(self.root / scene / p_type / file)
            ps = self.convert(pesudo_path)
            if self.use_prior:
                ps = ps * (mall+1)
            ps /= ps.max()
            p.append(ps)
        return p
        
    def __getitem__(self, i):
        file, scene = self.file_scene_list[i]
        if self.noise_type:
            img_path = str(self.root / scene / self.noise_type / file)
        else:
            img_path = str(self.root / scene / self.cam_subdir / file)
        img = self.convert(img_path, True)

        if self.mode == 'train':
            p = self.get_pseudos(scene, file)
            return img, p
        elif self.mode.startswith('val') or self.mode == 'test':
            gaze_path = str(self.root / scene / self.gaze_subdir / file)
            extensions = ['.png', '.jpg', '.jpeg']
            for ext in extensions:
                gaze_path = str(self.root / scene / self.gaze_subdir / file).rsplit('.', 1)[0] + ext
                if os.path.exists(gaze_path):
                    break
            gaze  = self.convert(gaze_path)
            return img, gaze
        
        elif self.mode =='infer':
            output_folder = str(self.root / scene / self.out_folder)
            os.makedirs(output_folder, exist_ok=True)
            out_test_path = str(self.root / scene / self.out_folder / file)
            return img, out_test_path
        elif self.mode == 'infer_mix':
            output_folder = str(self.root / scene / self.out_folder)
            os.makedirs(output_folder, exist_ok=True)
            out_test_path = str(self.root / scene / self.out_folder / file)
            p = self.get_pseudos(scene, file)
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
        sum_gaze = 0
        total_samples = 0
        for gazes in loader:
            gazes = gazes.cuda()
            batch_sum = gazes.sum(dim=0) 
            sum_gaze += batch_sum
            total_samples += gazes.size(0)  
                
        self.gaze_average = sum_gaze / total_samples
        out = self.gaze_average.permute(1, 2, 0).cpu().detach().numpy() * 255
        out = out.astype(np.uint8)
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
    



    def __len__(self):
        # return min(100, len(self.file_scene_list))
        return len(self.file_scene_list)

    @staticmethod
    def convert(x_path, is_rgb=False):
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
