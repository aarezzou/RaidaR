import os
import shutil

lst = os.listdir("./")
for fl in lst:
    if fl.endswith("g"):
        flder_name = fl[:-4]
        os.mkdir(flder_name)
        shutil.copyfile(fl, flder_name + '/' + fl)