#!/usr/bin/env python

from os import path, listdir, makedirs
from sys import argv
from tester import apply_model

def main():
    if len(argv) != 3:
        return
    
    working_dir = argv[1]
    
    for file in listdir(working_dir):
        file_name, file_ext = path.splitext(file)
        if file_ext == ".pt":
            model_name = f"{working_dir}/{file}"
            image_name = f"{working_dir}/{file_name.split('-nn-model')[0]}.png"
            output_dir = f"{working_dir}/{argv[2]}"
            makedirs(output_dir, exist_ok=True)
            
            apply_model(model_name, image_name, output_dir, True)
            
    
            
    

if __name__ == "__main__":
    main()