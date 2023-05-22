from torch.utils.data import Dataset
import random, os, pickle, torch
import numpy as np
from utils import RandomCrop, Compose, ToTensor, Normalize

lable_mapping = {'Preparation':0, 'CalotTriangleDissection':1, 'ClippingCutting':2,
         'GallbladderDissection':3, 'GallbladderPackaging':4, 'CleaningCoagulation':5,
         'GallbladderRetraction':6}

class load_trainset(Dataset):
    def __init__(self, file_paths, is_finetune, fps=25, fdim=1024):
        self.file_paths = file_paths
        self.fdim=fdim
        self.seq_len = (800, 7999)
        # self.seq_len = (1000, 6000)  # was good
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
            one_video = []
            for i in range(len(file_labels[id]['phase_gt_sec'])):
                one_video.append({'id':id, 'time':list(file_labels[id]['phase_gt_sec'].keys())[i], 'label':list(file_labels[id]['phase_gt_sec'].values())[i]})
                if list(file_labels[id]['phase_gt_sec'].values())[i] == 7:
                    print('attentin, bad label')
            train_list.append(one_video)
        print('train samples:', len(train_list))

        for id in test_id:
            one_video = []
            for i in range(len(file_labels[id]['phase_gt_sec'])):
                one_video.append({'id': id, 'time': list(file_labels[id]['phase_gt_sec'].keys())[i], 'label': list(file_labels[id]['phase_gt_sec'].values())[i]})
            val_list.append(one_video)
        print('valid samples:', len(val_list))
        return train_list, val_list

    def __getitem__(self, idx):
        id = self.train_list[idx]
        data_path = os.path.join(self.file_paths, id[0]['id'])
        anno = self.file_labels[id[0]['id']]
        # aug
        auglen = np.random.randint(500, 5200)
        vlen = len(id)
        if auglen > vlen:
            aug_frame_idx = random.sample(range(vlen*self.fps), auglen-vlen)
            aug_frame_idx = [i//self.fps for i in aug_frame_idx]
            aug_s_idx = sorted(aug_frame_idx+list(range(vlen)))
        else:
            drop_s_idx = random.sample(range(vlen), vlen-auglen)
            aug_s_idx = []
            for i in list(range(vlen)):
                if i in drop_s_idx:
                    continue
                else:
                    aug_s_idx.append(i)

        vdata = np.zeros((auglen, self.fdim))  #768
        phase_label, tool_label = [], []
        for i, idx in enumerate(aug_s_idx):
            fname = str(idx)+'.npy'
            if fname in os.listdir(data_path):
                v = np.load(os.path.join(data_path, fname))
                fidx = np.random.randint(0, 25)
                vdata[i,:] = v[fidx,:]
                phase_label.append(id[idx]['label'])
                if id[idx]['time']== 0:
                    tool_label.append([0,0,0,0,0,0,0])
                else:
                    tool_label.append(anno['tool_gt_sec'][id[idx]['time'] - 1])

            else:
                print('missing:',id[0]['id'],'@', i, 's')
                phase_label.append(phase_label[-1])
                tool_label.append(tool_label[-1])

        phase_label_m = torch.zeros((len(phase_label), 7))
        for i in range(len(phase_label)):
            phase_label_m[i, phase_label[i]] = 1
        phase_label = torch.tensor(phase_label)
        target = {'phase_label': phase_label, 'tool_label':torch.tensor(tool_label), 'phase_label_m':phase_label_m}
        samples = torch.FloatTensor(vdata)
        return samples, target

