
import os
import shutil

# basepath = '/home/bernard/develop/dataset/imagenet/train/'
basepath = '/workspace/develop/dataset/imagenet_1k/train/'
basepath_calib = '/workspace/develop/dataset/imagenet/calib/'

directory = os.fsencode(basepath)
if not os.path.exists(basepath_calib):
    os.mkdir(basepath_calib)
for d in os.listdir(directory):
    dir_name = os.fsdecode(d)
    dir_path = os.path.join(basepath,dir_name)
    dir_copy_path = os.path.join(basepath_calib,dir_name)
    if not os.path.exists(dir_copy_path):
        os.mkdir(dir_copy_path)
    for f in os.listdir(dir_path):
        file_path = os.path.join(dir_path,f)
        copy_file_path = os.path.join(dir_copy_path,f)
        # shutil.copyfile(file_path, copy_file_path)
        os.symlink(file_path, copy_file_path)
        break