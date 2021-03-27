import os
from PIL import Image

lst = os.listdir("./")
i = 0
for fl in lst:
    if fl.endswith("py"):
        continue
    if os.path.isdir("./" + fl):
        name = os.listdir("./" + fl)[0]
        path = "./" + fl + "/" + name
        if path.endswith("png"):
            print(i, path)
            img = Image.open(path)
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask = img.split()[3])
            background = background.resize((910, 512), Image.ANTIALIAS)
            background.save(path[:-3] + 'jpg', "JPEG", quality=100)
            os.remove(path)
            i = i+1

        