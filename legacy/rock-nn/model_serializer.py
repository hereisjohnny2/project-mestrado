#!/usr/bin/env python3
from sys import argv
from os import path
import torch
from rock_model import RockNetModel

def main():
    if len(argv) != 2:
        print("usage: ./model_serializer <path-to-model>")
        return
    
    model_path = argv[1]
    dirname, model_name = path.split(path.splitext(model_path)[0])
    
    output_name = f"{dirname}/{model_name}-scripted.pt"
    
    model = RockNetModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    scripted_model = torch.jit.script(model)
    scripted_model.save(output_name)


if __name__ == "__main__":
    main()