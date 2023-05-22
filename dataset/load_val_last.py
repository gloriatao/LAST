from torch.utils.data import Dataset
import os, pickle, torch
import numpy as np
from utils import RandomCrop, Compose, ToTensor, Normalize


lable_mapping = {'Preparation':0, 'CalotTriangleDissection':1, 'ClippingCutting':2,
         'GallbladderDissection':3, 'GallbladderPackaging':4, 'CleaningCoagulation':5,
         'GallbladderRetraction':6}

class load_valset(Dataset):
    def __init__(self, file_paths, is_test = False, fps=25, fdim=1024):
        self.file_paths = file_paths
        self.fps = fps  # 25fps to 1fps
        self.fdim=fdim
        with open('annotation_sec.pickle', 'rb') as f:  # run this to load from pickle
            self.file_labels = pickle.load(f)
        f.close()
        train_id, val_id = [], []
        
        if is_test:
            for i in range(80):
                if i < 40:
                    train_id.append('video' + "{0:02d}".format(i + 1))
                else:
                    val_id.append('video' + "{0:02d}".format(i + 1))
        else:
            for i in range(80):
                if i < 40:
                    train_id.append('video' + "{0:02d}".format(i + 1))
                if (i < 40) and (i >= 32):
                    val_id.append('video' + "{0:02d}".format(i + 1))
                    
                
                
                
        self.train_list, self.val_list = self.build_tran_val_list(train_id, val_id, self.file_labels)

        self.transform = Compose([RandomCrop(224),
                                  ToTensor(),
                                  Normalize([0.40577063, 0.27282622, 0.28533617],
                                            [0.24071056, 0.19952665, 0.20165241])
                               ])

    def __len__(self):
        return len(self.val_list)

    def build_tran_val_list(self, train_id, test_id, file_labels):
        train_list, val_list = [], []
        for id in test_id:
            one_video = []
            for i in range(len(file_labels[id]['phase_gt_sec'])):
                one_video.append({'id': id, 'time': list(file_labels[id]['phase_gt_sec'].keys())[i], 'label': list(file_labels[id]['phase_gt_sec'].values())[i]})
            val_list.append(one_video)
        print('valid samples:', len(val_list))
        return train_list, val_list

    def __getitem__(self, idx):
        id = self.val_list[idx]
        data_path = os.path.join(self.file_paths, id[0]['id'])
        anno = self.file_labels[id[0]['id']]

        vdata = np.zeros((len(id),self.fdim)) # 768
        phase_label, tool_label = [], []
        for i in range(len(id)):
            fname = str(i)+'.npy'
            if fname in os.listdir(data_path):
                v = np.load(os.path.join(data_path, fname))
                vdata[i,:] = v.mean(axis=0)

                phase_label.append(id[i]['label'])
                if id[i]['time']== 0:
                    tool_label.append([0,0,0,0,0,0,0])
                else:
                    tool_label.append(anno['tool_gt_sec'][id[i]['time'] - 1])

            else:
                print('missing:',id[0]['id'],'@', i, 's')
                phase_label.append(phase_label[-1])
                tool_label.append(tool_label[-1])

        phase_label_m = torch.zeros((len(phase_label), 7))
        for i in range(len(phase_label)):
            phase_label_m[i, phase_label[i]] = 1
        phase_label = torch.tensor(phase_label)
        target = {'phase_label': phase_label, 'tool_label':torch.tensor(tool_label), 'phase_label_m':phase_label_m, 'id':id[0]['id']}
        samples = torch.FloatTensor(vdata)
        return samples, target
