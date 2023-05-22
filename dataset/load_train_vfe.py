from torch.utils.data import Dataset
from PIL import Image
import random, os, pickle, torch
import numpy as np
from utils import RandomCrop, RandomHorizontalFlip, RandomRotation, ColorJitter, Compose, ToTensor, Normalize

lable_mapping = {'Preparation':0, 'CalotTriangleDissection':1, 'ClippingCutting':2,
         'GallbladderDissection':3, 'GallbladderPackaging':4, 'CleaningCoagulation':5,
         'GallbladderRetraction':6}

class load_trainset(Dataset):
    def __init__(self, file_paths, is_finetune=None, fps=25):
        self.file_paths = file_paths
        self.fps = fps  # 25fps to 1fps
        with open('annotation_sec.pickle', 'rb') as f:  # run this to load from pickle
            self.file_labels = pickle.load(f)
        f.close()
        train_id, val_id = [], []
        if is_finetune:
            for i in range(80):
                if i < 40:
                    train_id.append('video' + "{0:02d}".format(i + 1))
                if (i < 40) and (i >= 32):
                    val_id.append('video' + "{0:02d}".format(i + 1))
        else:
            for i in range(80):
                if i < 32:
                    train_id.append('video' + "{0:02d}".format(i + 1))
                if (i < 40) and (i >= 32):
                    val_id.append('video' + "{0:02d}".format(i + 1))
            

        self.train_list, self.val_list = self.build_tran_val_list(train_id, val_id, self.file_labels)

        self.transform = Compose([RandomCrop(224),
                                  ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.01),
                                  RandomHorizontalFlip(),
                                  RandomRotation(10),
                                  ToTensor(),
                                  Normalize([0.40577063, 0.27282622, 0.28533617],
                                            [0.24071056, 0.19952665, 0.20165241])
                               ])

    def __len__(self):
        return len(self.train_list)

    def build_tran_val_list(self, train_id, test_id, file_labels):
        rep = 4
        train_list, val_list = [], []
        for id in train_id:
            for i in range(len(file_labels[id]['phase_gt_sec'])):
                train_list.append({'id':id, 'time':list(file_labels[id]['phase_gt_sec'].keys())[i], 'label':list(file_labels[id]['phase_gt_sec'].values())[i]})
                if list(file_labels[id]['phase_gt_sec'].values())[i] == 7:
                    for tmp in range(rep): 
                        train_list.append({'id': id, 'time': list(file_labels[id]['phase_gt_sec'].keys())[i],'label': list(file_labels[id]['phase_gt_sec'].values())[i]})
        print('train samples:', len(train_list))

        for id in test_id:
            for i in range(len(file_labels[id]['phase_gt_sec'])):
                val_list.append({'id': id, 'time': list(file_labels[id]['phase_gt_sec'].keys())[i], 'label': list(file_labels[id]['phase_gt_sec'].values())[i]})
        print('valid samples:', len(val_list))
        return train_list, val_list

    def __getitem__(self, idx):
        id = self.train_list[idx]
        vdata = np.load(os.path.join(self.file_paths, id['id'],'%.6d.npy'%id['time']))
        fidx = np.random.randint(0, 25)
        image = vdata[fidx,:,:,:]

        seed = random.randint(0, 1e5)
        image = Image.fromarray(np.uint8(image))
        image = self.transform(image, seed)
        image = np.array(image)

        phase_label = id['label']

        anno = self.file_labels[id['id']]
        if id['time'] == 0:
            tool_label = [0,0,0,0,0,0,0]
        else:
            tool_label = anno['tool_gt_sec'][id['time'] - 1]

        target = {'phase_label':phase_label, 'tool_label':torch.tensor(tool_label)}
        samples = torch.tensor(image)
        return samples, target

