# -*- coding: utf-8 -*-
"""
@file      :  1130_thread_gen.py
@Time      :  2022/11/30 18:48
@Software  :  PyCharm
@summary   :
@Author    :  Bajian Xiang
"""

# nohup python thread_process.py  >> /data2/hsl/thread_0323_gen_data.log 2>&1 &

import datetime
import os
import threading


def execCmd(cmd):
    try:
        print("COMMAND -- %s -- BEGINS -- %s -- " % (cmd, datetime.datetime.now()))
        os.system(cmd)
        print("COMMAND -- %s -- ENDS -- %s -- " % (cmd, datetime.datetime.now()))
    except:
        print("Failed -- %s -- " % cmd)


# 如果只是路径变了的话，就改这3个地方
# Don't forget the last '/' in those paths!!!!
# Carefully check!!!

dir_str_head = "/data2/hsl/0323_wav_data/add_with_zky_0316/Speech/"  #
save_dir_head = "/data2/hsl/0323_pt_data/add_with_zky_0316/train/"
csv_path_head = "/data2/hsl/0323_wav_data/add_with_zky_0316/Speech/"  # 新传上去了

dir_str = [#dir_str_head + "arthur-sykes-rymer-auditorium-university-york",
           #dir_str_head + "creswell-crags",
           #dir_str_head + "elveden-hall-suffolk-england",
           #dir_str_head + "Cas-YanXiHu",
           dir_str_head + "Cas-zhongguancun",
           ]

save_dir = [#save_dir_head + "arthur-sykes-rymer-auditorium-university-york",
            #save_dir_head + "creswell-crags",
            #save_dir_head + "elveden-hall-suffolk-england",
            #save_dir_head + "Cas-YanXiHu",
            save_dir_head + "Cas-zhongguancun",
            ]

csv_dir = [#csv_path_head + "arthur-sykes-rymer-auditorium-university-york",
            #csv_path_head + "creswell-crags",
            #csv_path_head + "elveden-hall-suffolk-england",
            #csv_path_head + "Cas-YanXiHu"+"/20230321T150605_test_gen_corpus_dataset_results.csv",
            csv_path_head + "Cas-zhongguancun"+"/20230321T150605_test_gen_corpus_dataset_results.csv",
            ]


if __name__ == "__main__":
    commands = ["python 0921_OurData_GenPT.py --dir_str " + dir_str[i] + " --save_dir " + save_dir[i] + " --csv_file " + csv_dir[i] for i in range(len(dir_str))]
    threads = []
    for cmd in commands:
        th = threading.Thread(target=execCmd, args=(cmd,))
        th.start()
        threads.append(th)
    # 等待线程运行完毕
    for th in threads:
        th.join()

