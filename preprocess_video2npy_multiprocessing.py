import cv2
import os
import numpy as np
import multiprocessing

data_path = '/media/runze/DATA/Dasaset/cholec80/cholec80/videos'
out_png = '/media/runze/DATA/Dasaset/cholec80/cholec80/npys'

fps = 25
if not os.path.isdir(out_png):
    os.mkdir(out_png)

def process_video(f, out_png):
    id = f[:-4]
    out_dir = os.path.join(out_png, id)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    vidcap = cv2.VideoCapture(os.path.join(data_path, f))
    cap, frame = vidcap.read()

    if cap == False:
        print(f, ' cannot open video file')

    count, cnt, counts = 0, 0, 0
    vdata = np.zeros((fps, 250, 250, 3), dtype=np.uint8)

    while cap:
        if (count%fps==0):
            np.save(os.path.join(out_dir, '%.6d.npy'%counts), vdata)
            print(id, '%.6d.npy'%counts)
            counts += 1
            cnt = 0
            vdata = np.zeros((fps, 250, 250, 3), dtype=np.uint8)

        cap, frame = vidcap.read()
        if cap:
            frame = cv2.resize(frame, (250, 250), cv2.INTER_LINEAR)
            print('npy index:',cnt, 'frame index:',count, 'time index:',counts)
            vdata[cnt,:,:,:] = frame

            count += 1
            cnt += 1
        else:
            np.save(os.path.join(out_dir, '%.6d.npy'%counts), vdata)
            print('done',id, '%.6d.npy'%counts)
            break

# video06
pool = multiprocessing.Pool(8)
f = sorted(os.listdir(data_path))
pool.starmap(process_video, [(filename, out_png) for filename in sorted(f)])





