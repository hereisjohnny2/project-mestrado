import trainer
import tester
from utils.parser import create_parser
from utils.image import convert_image_to_rgb_format

def main(): 
    args = create_parser()
    
    try:
        if args.train:  
            epochs = int(args.epochs[0]) if args.epochs else 5
            trainer.train_from_dataset(args.train[0], epochs)
            return
        
        if args.image:
            convert_image_to_rgb_format(args.image[0])
        
        if args.image and args.model and args.output:
            tester.apply_model(args.model[0], args.image[0], args.output[0], args.save)
            return
            
    except TypeError as e:
         print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")

if __name__ == "__main__":
    main()