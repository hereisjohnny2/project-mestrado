import argparse

def create_parser():
    parser = argparse.ArgumentParser(description = "Rock NN CLI")
    
    parser.add_argument("-t", "--train", type = str, nargs = 1,
                        default = None,help = "Train a new model from dataset")
    
    parser.add_argument("-i", "--image", type = str, nargs = 1,
                        default = None, help = "Image to test")
    
    parser.add_argument("-m", "--model", type = str, nargs = 1,
                        default = None, help = "Model file location")
    
    parser.add_argument("-o", "--output", type = str, nargs = 1,
                        default = None, help = "Select the output folder")
    
    parser.add_argument("-e", "--epochs", type = int, nargs = 1,
                         default = None, help = "Number of epochs in training")
   
    parser.add_argument("-s", "--save", action='store_true',
                         default = False, help = "Save the result of binarization process")
    
    return parser.parse_args()