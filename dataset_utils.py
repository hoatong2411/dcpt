import os
import random
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
import torch
from utils.image_utils import random_augmentation, crop_img
from utils.degradation_utils import Degradation

REPEAT = 2

class TrainDataset(Dataset):
    def __init__(self, args):
        super(TrainDataset, self).__init__()
        self.args = args
        self.rs_ids = []
        self.hazy_ids = []
        self.D = Degradation(args)
        self.de_temp = 0
        self.de_type = self.args.de_type
        print(self.de_type)
        self.de_dict = {'denoise_15': 0, 
                        'denoise_25': 1, 
                        'denoise_50': 2, 
                        'derain': 3, 
                        'dehaze': 4, 
                        'deblur' : 5, 
                        'enhance' : 6, 
                        'desnow': 7}
        
        self._init_ids()
        self._merge_ids()

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])
        self.toTensor = ToTensor()

    def _init_ids(self):
        if 'denoise_15' in self.de_type or 'denoise_25' in self.de_type or 'denoise_50' in self.de_type:
            self._init_clean_ids()
        if 'derain' in self.de_type:
            self._init_derain_ids()
        if 'dehaze' in self.de_type:
            self._init_dehaze_ids()
        if 'deblur' in self.de_type:
            self._init_deblur_ids()
        if 'enhance' in self.de_type:
            self._init_enhance_ids()
        if 'desnow' in self.de_type:
            self._init_desnow_ids()
        random.shuffle(self.de_type)

    def _init_clean_ids(self):
        """
        ref_file = self.args.data_file_dir + "noisy/denoise.txt"
        temp_ids = []
        temp_ids+= [id_.strip() for id_ in open(ref_file)]
        clean_ids = []
        name_list = os.listdir(self.args.denoise_dir)
        clean_ids += [self.args.denoise_dir + id_ for id_ in name_list if id_.strip() in temp_ids]

        if 'denoise_15' in self.de_type:
            self.s15_ids = [{"clean_id": x,"de_type":0} for x in clean_ids]
            self.s15_ids = self.s15_ids * REPEAT
            random.shuffle(self.s15_ids)

        if 'denoise_25' in self.de_type:
            self.s25_ids = [{"clean_id": x,"de_type":1} for x in clean_ids]
            self.s25_ids = self.s25_ids * REPEAT
            random.shuffle(self.s25_ids)

        if 'denoise_50' in self.de_type:
            self.s50_ids = [{"clean_id": x,"de_type":2} for x in clean_ids]
            self.s50_ids = self.s50_ids * REPEAT
            random.shuffle(self.s50_ids)

        self.num_clean = len(clean_ids)
        print("Total Denoise Ids : {}".format(self.num_clean))
        """
        pass 


    def _list_lq(self, task_dir):
        lq_dir = os.path.join(task_dir, 'LQ')
        return sorted(os.listdir(lq_dir))

    def _init_dehaze_ids(self):
        file_list = self._list_lq(self.args.dehaze_dir)
        self.dehaze_ids = [{"clean_id": f, "de_type": 4} for f in file_list] * REPEAT
        print("Total Dehaze Ids (raw): {} | after repeat: {}".format(
            len(file_list), len(self.dehaze_ids)))
        random.shuffle(self.dehaze_ids)

    def _init_deblur_ids(self):
        file_list = self._list_lq(self.args.deblur_dir)
        self.deblur_ids = [{"clean_id": f, "de_type": 5} for f in file_list] * REPEAT
        print("Total Deblur Ids (raw): {} | after repeat: {}".format(
            len(file_list), len(self.deblur_ids)))
        random.shuffle(self.deblur_ids)

    def _init_enhance_ids(self):
        file_list = self._list_lq(self.args.enhance_dir)
        self.enhance_ids = [{"clean_id": f, "de_type": 6} for f in file_list] * REPEAT
        print("Total Enhance Ids (raw): {} | after repeat: {}".format(
            len(file_list), len(self.enhance_ids)))
        random.shuffle(self.enhance_ids)

    def _init_derain_ids(self):
        file_list = self._list_lq(self.args.derain_dir)
        self.derain_ids = [{"clean_id": f, "de_type": 3} for f in file_list] * REPEAT
        print("Total Derain Ids (raw): {} | after repeat: {}".format(
            len(file_list), len(self.derain_ids)))
        random.shuffle(self.derain_ids)

    def _init_desnow_ids(self):
        file_list = self._list_lq(self.args.desnow_dir)
        self.desnow_ids = [{"clean_id": f, "de_type": 7} for f in file_list] * REPEAT
        print("Total Desnow Ids (raw): {} | after repeat: {}".format(
            len(file_list), len(self.desnow_ids)))
        random.shuffle(self.desnow_ids)
    

    def _merge_ids(self):
        self.sample_ids = []
        if 'denoise_15' in self.de_type:
            self.sample_ids += self.s15_ids
        if 'denoise_25' in self.de_type:
            self.sample_ids += self.s25_ids
        if 'denoise_50' in self.de_type:
            self.sample_ids += self.s50_ids
        if 'derain' in self.de_type:
            self.sample_ids += self.derain_ids
        if 'dehaze' in self.de_type:
            self.sample_ids += self.dehaze_ids
        if 'deblur' in self.de_type:
            self.sample_ids += self.deblur_ids
        if 'enhance' in self.de_type:
            self.sample_ids += self.enhance_ids
        if 'desnow' in self.de_type:
            self.sample_ids += self.desnow_ids
        random.shuffle(self.sample_ids)
        print("Total merged samples: {}".format(len(self.sample_ids))) 


    def _load_lq_gt(self, task_dir, filename):
        """Load images from <task_dir>/LQ/<filename> and <task_dir>/GT/<filename>."""
        lq_path = os.path.join(task_dir, 'LQ', filename)
        gt_path = os.path.join(task_dir, 'GT', filename)
        lq_img = crop_img(np.array(Image.open(lq_path).convert('RGB')), base=16)
        gt_img = crop_img(np.array(Image.open(gt_path).convert('RGB')), base=16)
        return lq_img, gt_img
    
    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        de_id = sample["de_type"]

        if de_id < 3:
            # Denoise
            clean_id  = sample["clean_id"]
            clean_img = crop_img(np.array(Image.open(clean_id).convert('RGB')), base=16)
            clean_patch = self.crop_transform(clean_img)
            clean_patch= np.array(clean_patch)
            clean_name = clean_id.split("/")[-1].split('.')[0]
            clean_patch = random_augmentation(clean_patch)[0]
            degrad_patch = self.D.single_degrade(clean_patch, de_id)

        else:
            filename = sample["clean_id"]

            if de_id == 3:
                # Rain Streak Removal
                degrad_img, clean_img = self._load_lq_gt(self.args.derain_dir, filename)
            elif de_id == 4:
                # Dehazing with SOTS outdoor training set
                degrad_img, clean_img = self._load_lq_gt(self.args.dehaze_dir, filename)
            elif de_id == 5:
                # Deblur with Gopro set
                degrad_img, clean_img = self._load_lq_gt(self.args.deblur_dir, filename)
            elif de_id == 6:
                # Enhancement with LOL training set
                degrad_img, clean_img = self._load_lq_gt(self.args.enhance_dir, filename)
            elif de_id == 7:
                degrad_img, clean_img = self._load_lq_gt(self.args.desnow_dir, filename)

            clean_name = os.path.splitext(filename)[0]   
            degrad_patch, clean_patch = random_augmentation(*self._crop_patch(degrad_img, clean_img))

        clean_patch = self.toTensor(clean_patch)
        degrad_patch = self.toTensor(degrad_patch)
        return [clean_name, de_id], degrad_patch, clean_patch
    
    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)
        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        return patch_1, patch_2

    def __len__(self):
        return len(self.sample_ids)

class DenoiseTestDataset(Dataset):
    def __init__(self, args):
        super(DenoiseTestDataset, self).__init__()
        self.args = args
        self.clean_ids = []
        self.sigma = 15
        self._init_clean_ids()
        self.toTensor = ToTensor()

    def _init_clean_ids(self):
        name_list = os.listdir(self.args.denoise_path)
        self.clean_ids += [self.args.denoise_path + id_ for id_ in name_list]
        self.num_clean = len(self.clean_ids)

    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def set_sigma(self, sigma):
        self.sigma = sigma

    def __getitem__(self, clean_id):
        clean_img = crop_img(np.array(Image.open(self.clean_ids[clean_id]).convert('RGB')), base=16)
        clean_name = self.clean_ids[clean_id].split("/")[-1].split('.')[0]
        noisy_img, _ = self._add_gaussian_noise(clean_img)
        clean_img, noisy_img = self.toTensor(clean_img), self.toTensor(noisy_img)
        return [clean_name], noisy_img, clean_img

    def tile_degrad(input_,tile=128,tile_overlap =0):
        sigma_dict = {0:0,1:15,2:25,3:50}
        b, c, h, w = input_.shape
        tile = min(tile, h, w)
        assert tile % 8 == 0, "tile size should be multiple of 8"
        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h, w).type_as(input_)
        W = torch.zeros_like(E)
        s = 0
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = input_[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = in_patch
                out_patch_mask = torch.ones_like(in_patch)

                E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
                W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
        restored = torch.clamp(restored, 0, 1)
        return restored
    def __len__(self):
        return self.num_clean


class DerainDehazeDataset(Dataset):
    def __init__(self, args, task="derain",addnoise = False,sigma = None):
        super(DerainDehazeDataset, self).__init__()
        self.ids = []
        self.task_idx = 0
        self.args = args
        self.task_dict = {
            'derain': 0, 
            'dehaze': 1, 
            'deblur': 2, 
            'enhance': 3, 
            'desnow': 4
            }
        self.path_map = {
            0: args.derain_path,
            1: args.dehaze_path,
            2: args.gopro_path,
            3: args.enhance_path,
            4: args.desnow_path,
        }

        self.toTensor = ToTensor()
        self.addnoise = addnoise
        self.sigma = sigma
        self.set_dataset(task)

    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def _init_input_ids(self):
        base_path = self.path_map[self.task_idx]
        lq_dir = os.path.join(base_path, 'LQ')
        self.ids = [os.path.join(lq_dir, f) for f in sorted(os.listdir(lq_dir))]
        self.length = len(self.ids)
        print("Test task_idx={} | {} samples".format(self.task_idx, self.length))

    def _get_gt_path(self, lq_path):
        return lq_path.replace(os.sep + 'LQ' + os.sep, os.sep + 'GT' + os.sep)

    def set_dataset(self, task):
        self.task_idx = self.task_dict[task]
        self._init_input_ids()

    def __getitem__(self, idx):
        degraded_path = self.ids[idx]
        clean_path = self._get_gt_path(degraded_path)

        degraded_img = crop_img(np.array(Image.open(degraded_path).convert('RGB')), base=16)
        if self.addnoise:
            degraded_img,_ = self._add_gaussian_noise(degraded_img)
        clean_img = crop_img(np.array(Image.open(clean_path).convert('RGB')), base=16)
        clean_img, degraded_img = self.toTensor(clean_img), self.toTensor(degraded_img)
        degraded_name = degraded_path.split('/')[-1][:-4]
        return [degraded_name], degraded_img, clean_img

    def __len__(self):
        return self.length

class TestSpecificDataset(Dataset):
    def __init__(self, args):
        super(TestSpecificDataset, self).__init__()
        self.args = args
        self.degraded_ids = []
        self._init_clean_ids(args.test_path)
        self.toTensor = ToTensor()

    def _init_clean_ids(self, root):
        extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']
        if os.path.isdir(root):
            name_list = []
            for image_file in os.listdir(root):
                if any([image_file.endswith(ext) for ext in extensions]):
                    name_list.append(image_file)
            if len(name_list) == 0:
                raise Exception('The input directory does not contain any image files')
            self.degraded_ids += [root + id_ for id_ in name_list]
        else:
            if any([root.endswith(ext) for ext in extensions]):
                name_list = [root]
            else:
                raise Exception('Please pass an Image file')
            self.degraded_ids = name_list
        print("Total Images : {}".format(name_list))
        self.num_img = len(self.degraded_ids)

    def __getitem__(self, idx):
        degraded_img = crop_img(np.array(Image.open(self.degraded_ids[idx]).convert('RGB')), base=16)
        name = self.degraded_ids[idx].split('/')[-1][:-4]
        degraded_img = self.toTensor(degraded_img)
        return [name], degraded_img

    def __len__(self):
        return self.num_img
    
