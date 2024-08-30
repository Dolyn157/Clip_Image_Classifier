import os
from PIL import Image


allfiles = []

def get_all_file(rawdir):
    allfilelist = os.listdir(rawdir)
    for f in allfilelist:
        filepath = os.path.join(rawdir, f)
        if os.path.isdir(filepath):
            get_all_file(filepath)
        allfiles.append(filepath)
    return allfiles

def get_first_level_file(rawdir):
    allfilelist = os.listdir(rawdir)
    for f in allfilelist:
        filepath = os.path.join(rawdir, f)
        if os.path.isfile(filepath):
            allfiles.append(filepath)
    return allfiles

def is_valid_image(filename):
    try:
        img = Image.open(filename)
        img.verify()
        return True
    except (IOError, SyntaxError):
        return False


