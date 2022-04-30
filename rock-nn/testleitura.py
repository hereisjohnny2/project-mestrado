#!/usr/bin/env python3

from tester import apply_model

def main():
    full_path = "/home/joao/projects/python/project-mestrado"
    apply_model(
        model_file=f"{full_path}/images/Berea200/it-model-per-image/I22-nn-model.pt", 
        image=f"{full_path}/images/Berea200/I22.png", 
        output=f"{full_path}/images/Berea200/test",
        save=True)

if __name__ == "__main__":
    main()