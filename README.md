## **T60_data_generation**

### Code Description
  0921_OurData_GenPT.py is used to generate *.pt from *.wav, which calls the fuunction from gen_specgram.py
  if you want to generate your own *.pt from your own *.wav, you should revise the following places:
    default in parser.add_argument('--csv_file', type=str, default= "/data2/pyq/yousonic/code_split_wav/gt_zky_t60.csv) by using your own t60_groundtruth file.
    default in parser.add_argument('--dir_str_head', type=str, default="/data2/pyq/yousonic/20230214_zky/") by using your own path which is storing *.wav files.
    default in parser.add_argument('--save_dir'...) represents the path which your generated *.pt will store.
### Generated *.pt Description
  The type of *.pt is dict, and its size is 1. Its Key is its filename, and its value is a list. The size of the list is the number of slices, depended on the length of your *.wav. Each list is a dict, with the size of 4, including image, ddr, t60, MeanT60 and their corresponding values.
### Instructions
  if you want use the code to generate *.pt, you have better to run 0921_OurData_GenPT.pywhth gen_specgram.py, as well as your *.wav files. You may revise some variables in 0921_OurData_GenPT.py, just modifying according to the code conmments.
