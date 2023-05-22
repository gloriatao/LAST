from torch.utils.data import Dataset
from PIL import Image
import random, os, cv2, pickle, torch
import numpy as np
from utils import RandomCrop, RandomHorizontalFlip, RandomRotation, ColorJitter, Compose, ToTensor, Normalize, CenterCrop

lable_mapping = {'Preparation':0, 'CalotTriangleDissection':1, 'ClippingCutting':2,
         'GallbladderDissection':3, 'GallbladderPackaging':4, 'CleaningCoagulation':5,
         'GallbladderRetraction':6}

class load_valset(Dataset):
    def __init__(self, file_paths, fps=25):
        self.file_paths = file_paths
        self.fps = fps  # 25fps to 1fps
        # self.file_labels = self.load_label_from_txt(label_paths)  # run this to load from txt labels
        with open('annotation_sec.pickle', 'rb') as f:  # run this to load from pickle
            self.file_labels = pickle.load(f)
        f.close()
        train_id, val_id, test_id = [], [], []
        for i in range(80):
            if i < 49:
                train_id.append('video' + "{0:02d}".format(i + 1))
            else:
                test_id.append('video' + "{0:02d}".format(i + 1))

        self.all_list = self.build_val_list(train_id+test_id, self.file_labels)

        self.transform = Compose([CenterCrop(224),
                                  ToTensor(),
                                  Normalize([0.40577063, 0.27282622, 0.28533617],
                                            [0.24071056, 0.19952665, 0.20165241])
                               ])

    def __len__(self):
        return len(self.all_list)

    def build_val_list(self, test_id, file_labels):
        val_list = []
        for id in test_id:
            for i in range(len(file_labels[id]['phase_gt_sec'])):
                val_list.append({'id': id, 'time': list(file_labels[id]['phase_gt_sec'].keys())[i], 'label': list(file_labels[id]['phase_gt_sec'].values())[i]})
        print('val samples:', len(val_list))
        return val_list

    def __getitem__(self, idx):
        id = self.all_list[idx]
        vdata = np.load(os.path.join(self.file_paths, id['id'],'%.6d.npy'%id['time']))
        # frame = image.copy()
        # frame = frame-np.min(frame)
        # frame = frame/np.max(frame)
        # cv2.imwrite('/data/rtao/projects/cholec/debug/tmp.png', frame*255)
        seed = random.randint(0, 1e5)
        image = np.zeros((self.fps, 3, 224, 224))
        for i in range(self.fps):
            v = vdata[i,:,:,:]
            v = Image.fromarray(np.uint8(v))
            v = self.transform(v, seed)
            image[i,:,:,:] = np.array(v)

        # frame = image.copy()
        # frame = frame-np.min(frame)
        # frame = frame/np.max(frame)
        # cv2.imwrite('/data/rtao/projects/cholec/debug/tmp1.png', frame.transpose(1,2,0)*255)

        phase_label = id['label']

        anno = self.file_labels[id['id']]
        if id['time'] == 0:
            tool_label = [0,0,0,0,0,0,0]
        else:
            tool_label = anno['tool_gt_sec'][id['time'] - 1]

        target = {'phase_label':phase_label, 'tool_label':torch.tensor(tool_label), 'id':id}
        samples = torch.FloatTensor(image)
        return samples, target
#
# def test():
#     file_paths = '/data/rtao/data/cholec80/videos'
#     label_paths = '/data/rtao/data/cholec80/phase_annotations'
#     dataset = CholecDataset(file_paths, label_paths, seq_count=5)
#
#     _ = dataset.__getitem__(0)
#
# test()