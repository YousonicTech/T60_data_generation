# -*- coding: utf-8 -*-
"""
@file      :  0902_add_clean_to_pt.py
@Time      :  2022/9/2 14:26
@Software  :  PyCharm
@summary   :  读pt，删掉1kHz之外的其他倍频程，然后加入clean的语谱图
@Author    :  Bajian Xiang
"""
import os
import re
import glob
import torch
import numpy

clean_speech_path = "/data2/hsl/0324_clean_pt/*.pt"
original_pt_path = "/data2/hsl/0323_pt_data/add_with_zky_0316/train/Cas-YanXiHu/"  # 这里面又有train又有val，还得弄一下
new_pt_path = "/data2/hsl/0324_pt_with_clean/Cas-YanXiHu/"
# 对应1kHz的语谱图
which_freq = 2


def merge_clean_pt(clean_speech_path, which_freq):
    """
    clean_dict:{
        '东北话男声_1': clean spectrogram of 1kHz -- [slice1, slice2, ...]
        '东北话男声_2': clean spectrogram of 1kHz -- [slice1, slice2, ...]
        ...
    }
    """
    clean_dict = {}
    clean_names = glob.glob(clean_speech_path)
    for i in clean_names:
        temp_name = i.split('/')[-1].split('.')[0]
        temp_pt = torch.load(i)
        for key, value in temp_pt.items():
            temp_specgram_list = []
            temp_specgram_list.clear()
            for slice_num in range(len(value)):
                #temp_specgram_list.append(value[slice_num]['image'][which_freq])
                temp_specgram_list.append(value[slice_num]['image'])
        clean_dict[temp_name] = temp_specgram_list
    return clean_dict


def original_pt_add_clean(origin):
    for key, value in origin.items():
        # key:'creswell-crags_2_s_mainlevel_r_mouth_2_creswell-crags_四川话女声_1_TIMIT_a001_50_60_0dB-0'
        utter_name = re.findall(r'[\u4e00-\u9fa5]+_\d', key)[0]  # '四川话女声_1'
        clean_pt = clean_dict[utter_name]
        for slice in range(len(value)):
            #origin[key][slice]['image'] = origin[key][slice]['image'][which_freq]
            origin[key][slice]['image'] = origin[key][slice]['image']
            origin[key][slice]['clean'] = clean_pt[slice]
        return origin


if __name__ == '__main__':
    clean_dict = merge_clean_pt(clean_speech_path, which_freq)

    # for fir_dir in os.listdir(original_pt_path):
    #     # train, val
    #     temp_dir = os.path.join(original_pt_path, fir_dir)  # '/data/xbj/0831_1000hz/train'
    #     new_temp_dir = os.path.join(new_pt_path, fir_dir)   # '/data/xbj/0902_1000hz_with_clean/train'
    #     print('* current fir_dir: ', temp_dir)
    #     if not os.path.exists(new_temp_dir):
    #         print('* make new dir: ', new_temp_dir)
    #         os.makedirs(new_temp_dir)

    # for config_dir in os.listdir(original_pt_path):
    #     #temp_config_dir = os.path.join(temp_dir, config_dir)  # '/data/xbj/0831_1000hz/train/creswell-crags'
    #     #temp_config_new_dir = os.path.join(new_temp_dir, config_dir)  # '/data/xbj/0831_1000hz/train/creswell-crags'
    #     temp_config_dir = original_pt_path
    #     temp_config_new_dir = new_pt_path

    all_pt = glob.glob(original_pt_path + '/*.pt')
    for each_pt in all_pt:
        # each_pt : '/data/xbj/0831_1000hz/train/creswell-crags/creswell-crags_2_s_mainlevel_r_mouth_2_creswell-crags_四川话女声_1_TIMIT_a001_50_60_0dB-0.pt'
        temp_pt = torch.load(each_pt)
        temp_pt_add_clean = original_pt_add_clean(temp_pt)
        temp_pt_new_path = os.path.join(new_pt_path, each_pt.split(original_pt_path)[-1])
        torch.save(temp_pt_add_clean, temp_pt_new_path)
        print('----- save: ', temp_pt_new_path)


    print("--------------------finish all-------------------------")

