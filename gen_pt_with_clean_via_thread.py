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



clean_speech_path = "/data2/hsl/0324_clean_pt/"
original_pt_root = "/data2/hsl/0323_pt_data/add_without_zky_0316/train"
original_sub_dir  = ["central-hall-university-york","york-guildhall-council-chamber",'dixon-studio-theatre-university-york','hoffmann-lime-kiln-langcliffeuk','gill-heads-mine','koli-national-park-winter','elveden-hall-suffolk-england','ron-cooke-hub-university-york','creswell-crags','arthur-sykes-rymer-auditorium-university-york','koli-national-park-summer','innocent-railway-tunnel']
new_pt_root = "/data2/hsl/0324_pt_with_clean" 
print("a",original_pt_root)
print("b",original_sub_dir[0])
#original_pt_paths  = [print(os.path.join(original_pt_root,sub)) for sub in original_sub_dir]
original_pt_paths  = [os.path.join(original_pt_root,sub) for sub in original_sub_dir]
new_pt_paths = [os.path.join(new_pt_root,sub) for sub in original_sub_dir]
#new_pt_paths = [print(new_pt_root,original_sub_dir) for sub in original_sub_dir]

if __name__ == "__main__":
    commands = ["python 0927_add_clean_to_pt_args.py  --clean_speech_path " + clean_speech_path + " --original_pt_path " + original_pt_paths[i] + " --new_pt_path " + new_pt_paths[i] for i in range(len(original_pt_paths))]
    threads = []
    for cmd in commands:
        th = threading.Thread(target=execCmd, args=(cmd,))
        th.start()
        threads.append(th)
    # 等待线程运行完毕
    for th in threads:
        th.join()
