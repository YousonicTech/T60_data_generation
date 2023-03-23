# -*- coding: utf-8 -*-
"""
@file      :  0921_OurData_GenPT.py
@Time      :  2022/9/21 10:26
@Software  :  PyCharm
@summary   :  测了一批君正什么什么的数据，要生成一下相应的wav拿去训练，最好还能看看解混响的结果
@Author    :  Bajian Xiang
"""

import numpy as np
import splweighting
import wave
import glob
import os
import torch
import pandas as pd
import argparse
from gen_specgram import Filter_Downsample_Spec

##configuration area
chunk_length = 4
chunk_overlap = 0.5
# TODO 需要改变csv,numpydir/csv_save_path这些路径


parser = argparse.ArgumentParser(description='manual to this script')
# 切分用倍频程
parser.add_argument('--csv_file', type=str,
                    default="/data2/pyq/yousonic/code_split_wav/gt_zky_t60.csv")  # 中科院数据的t60文件
parser.add_argument('--dir_str_head', type=str,
                    default="/data2/pyq/yousonic/20230214_zky/")
parser.add_argument('--save_dir', type=str,
                    default="/data2/pyq/0323_ZKY_DATA_FOR_TEST/Test_zky_pt")

args = parser.parse_args()
save_dir = args.save_dir
if not os.path.exists(args.save_dir):
    os.makedirs(save_dir)
save_dir = args.save_dir
dir_str_head = args.dir_str_head
csv_file = args.csv_file
dir_str_ = [dir_str_head + "N106",
            dir_str_head + "N208",
            dir_str_head + "N306",
            dir_str_head + "N308",
            dir_str_head + "S102",
            dir_str_head + "S106",
            dir_str_head + "S201",
            dir_str_head + "S204",
            dir_str_head + "S206",
            dir_str_head + "S301",
            ]


##functions
def SPLCal(x):
    Leng = len(x)
    pa = np.sqrt(np.sum(np.power(x, 2)) / Leng)
    p0 = 2e-5
    spl = 20 * np.log10(pa / p0)
    return spl


class Totensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, ddr, t60, meanT60 = sample['image'], sample['ddr'], sample['t60'], sample['MeanT60']

        # image, ddr, t60 = sample['image'], sample['ddr'], sample['t60']
        image = [torch.from_numpy(m.astype(float)) for m in image]
        ddr = ddr.astype(float)
        t60 = t60.astype(float)
        meanT60 = meanT60.astype(float)
        # image = image.transpose((2, 0, 1))
        return {'image': image,
                'ddr': torch.from_numpy(ddr),
                't60': torch.from_numpy(t60),
                "MeanT60": torch.from_numpy(meanT60)
                }
        #


csv_data = pd.read_csv(csv_file)
# print(csv_data)
num = 0
# csv_data = pd.read_csv()
for i in range(len(dir_str_)):
    temp1 = os.listdir(dir_str_[i])
    for j in range(len(temp1)):
        dir_str = []
        dir_str.append((dir_str_[i] + '/' + temp1[j]))
        print(dir_str)

        # for wav_file_name in glob.glob(dir_str+r"/*.wav"):
        for file_name in glob.glob(dir_str[0] + r"/*.wav"):
            # wav_file_name = glob.glob(dir_str + r"/*.wav")[0]
            f = wave.open(file_name, "rb")
            params = f.getparams()
            nchannels, sampwidth, framerate, nframes = params[:4]
            # print(file_name)
            # print(nchannels, sampwidth, framerate, nframes / framerate)

            str_data = f.readframes(nframes)
            f.close()
            wave_data = np.frombuffer(str_data, dtype=np.int16)

            wave_data.shape = -1, nchannels
            wave_data = wave_data.T
            audio_time = nframes / framerate
            chan_num = 0
            count = 0
            new_file_name = (file_name.split("\\")[-1]).split(".")[0]
            new_file_name = new_file_name.split("/")[-3] + '-' + new_file_name.split("/")[-2]
            ## process each channel of audio
            for audio_samples_np in wave_data:
                whole_audio_SPL = SPLCal(audio_samples_np)

                available_part_num = (audio_time - chunk_overlap) // (
                        chunk_length - chunk_overlap)  # 4*x - (x-1)*0.5 <= audio_time    x为available_part_num

                if available_part_num == 1:
                    cut_parameters = [chunk_length]
                else:
                    cut_parameters = np.arange(chunk_length,
                                               (chunk_length - chunk_overlap) * available_part_num + chunk_overlap,
                                               chunk_length)  # np.arange()函数第一个参数为起点，第二个参数为终点，第三个参数为步长（10秒）

                start_time = int(0)  # 开始时间设为0
                count = 0
                # 开始存储pt文件
                dict = {}
                save_data = []
                for t in cut_parameters:
                    stop_time = int(t)  # pydub以毫秒为单位工作
                    start = int(start_time * framerate)
                    end = int((start_time + chunk_length) * framerate)
                    audio_chunk = audio_samples_np[start:end]  # 音频切割按开始时间到结束时间切割

                    ##ingore chunks with no audio
                    chunk_spl = SPLCal(audio_chunk)
                    if whole_audio_SPL - chunk_spl >= 20:
                        continue

                    # file naming

                    count += 1

                    # A weighting
                    chunk_a_weighting = splweighting.weight_signal(audio_chunk, framerate)

                    # gammatone
                    chunk_result, _, _ = Filter_Downsample_Spec(chunk_a_weighting, framerate)

                    chan = chan_num + 1
                    room = new_file_name

                    a = (csv_data['Room:'] == room)
                    # b = (csv_data['Room Config:'] == config).values

                    data = csv_data[a]
                    #  Freq: (Freq, T60, Mean T60, DRR, Mean DRR)
                    # 500Hz: (13, 13, 14, 13, 14)
                    # 1kHz:  (16, 16, 17, 16, 17)
                    # 2kHz:  (19, 19, 20, 19, 20)
                    # 4kHz:  (22, 22, 23, 22, 23)
                    T60_data = data.loc[:, ['T60:.13']]    # 13表示的是采样频率是500Hz对应的T60
                    FB_T60_data = data.loc[:, ['FB T60:']]
                    T60_M_data = data.loc[:, ['Mean T60:.14']]
                    DDR_each_band = np.array([0 for i in range(30)])
                    T60_each_band = (T60_data.values).reshape(-1)
                    MeanT60_each_band = np.array([FB_T60_data, T60_M_data])
                    image = chunk_result
                    # print('-- save image shape:', [x.shape for x in image], ' --')
                    sample = {'image': image, 'ddr': DDR_each_band, 't60': T60_each_band, "MeanT60": MeanT60_each_band}
                    transform = Totensor()
                    sample = transform(sample)

                    save_data.append(sample)

                    # 这里加上干净的pt

                    start_time = start_time + chunk_length - chunk_overlap  # 开始时间变为结束时间前1s---------也就是叠加上一段音频末尾的4s
                for slice_number in range(61):
                    save_data

                if len(save_data) != 0:
                    pt_file_name = os.path.join(save_dir, new_file_name + '-' + str(chan_num) + '.pt')
                    dict[new_file_name + '-' + str(chan_num)] = save_data
                    torch.save(dict, pt_file_name)

                chan_num = chan_num + 1
            print('----------------finish----------------')
