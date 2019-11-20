
from argparse import ArgumentParser
from decodeDependencies import decode

if __name__ == '__main__':
    
    arg_parser = ArgumentParser()

    arg_parser.add_argument("--input",
                            help="Path to the TSV file encoding the tree", 
                            default=None)
    arg_parser.add_argument("--output",
                            help="Path where to print the output file")
    
    args = arg_parser.parse_args()
    

    with open(args.input) as f:
        
        lines = f.readlines()
        
        print (decode(lines, args.output, "@","English"))