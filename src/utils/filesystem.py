import os
from pathlib import Path
import glob
from typing import Union
import zipfile

def get_audio_files(wd):
    files = glob.glob("{0}**/*.wav".format(wd), recursive=True)
    return files


def ensure_dir(directory: Union[str, Path]):
    """
    Ensure all directories along given path exist, given directory name
    """
    directory = str(directory)
    if len(directory) > 0 and not os.path.exists(directory):
        os.makedirs(directory)


def ensure_dir_for_filename(filename: str):
    """
    Ensure all directories along given path exist, given filename
    """
    ensure_dir(os.path.dirname(filename))

def make_zipfile(output_filename, source_dir):
    rel_root = os.path.abspath(os.path.join(source_dir, os.pardir))
    with zipfile.ZipFile(output_filename, "w", zipfile.ZIP_DEFLATED) as zip:
        for root, dirs, files in os.walk(source_dir):
            # add directory (needed for empty dirs)
            zip.write(root, os.path.relpath(root, rel_root))
            for file in files:
                filename = os.path.join(root, file)
                if os.path.isfile(filename): # regular files only
                    arc_name = os.path.join(os.path.relpath(root, rel_root), file)
                    zip.write(filename, arc_name)