from torch.utils.data import Dataset
from PIL import Image
import random, os, cv2, pickle, torch
import numpy as np
from utils import Compose, ToTensor, Normalize, CenterCrop

lable_mapping = {'Preparation':0, 'CalotTriangleDissection':1, 'ClippingCutting':2,
         'GallbladderDissection':3, 'GallbladderPackaging':4, 'CleaningCoagulation':5,
         'GallbladderRetraction':6}

class load_valset(Dataset):
    def __init__(self, file_paths, file_id, fps=25,):
        self.file_paths = file_paths
        self.fps = fps  
        with open('annotation_sec.pickle', 'rb') as f:  
            self.file_labels = pickle.load(f)
        f.close()
        self.test_list = self.build_val_list([file_id], self.file_labels)
        self.transform = Compose([CenterCrop(224),
                                  ToTensor(),
                                  Normalize([0.40577063, 0.27282622, 0.28533617],
                                            [0.24071056, 0.19952665, 0.20165241])
                               ])

    def __len__(self):
        return len(self.test_list)

    def build_val_list(self, test_id, file_labels):
        val_list = []
        for id in test_id:
            for i in range(len(file_labels[id]['phase_gt_sec'])):
                val_list.append({'id': id, 'time': list(file_labels[id]['phase_gt_sec'].keys())[i], 'label': list(file_labels[id]['phase_gt_sec'].values())[i]})
        print('val samples:', len(val_list))
        return val_list

    def __getitem__(self, idx):
        id = self.test_list[idx]
        vdata = np.load(os.path.join(self.file_paths, id['id'],'%.6d.npy'%id['time']))
        image = vdata[0,:,:,:]

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
        target = {'phase_label':phase_label, 'tool_label':torch.tensor(tool_label), 'id':id}
        samples = torch.FloatTensor(image)
        return samples, target
