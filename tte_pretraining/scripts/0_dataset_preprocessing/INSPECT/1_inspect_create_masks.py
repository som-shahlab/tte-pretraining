import argparse
from pe.datatasets.data_preprocessers.inspect import  InspectPreprocessing
import os 

    #main(args)
def main(args):
     inspect_preprocessor  = InspectPreprocessing(dataset_path=args.dataset_path)
     #data_dir              = inspect_preprocessor.process_files_parallel()
     #data_dir              = inspect_preprocessor.process_files()
     inspect_preprocessor.segment_studies(save_to=args.output_path,
                                          start   = args.start,
                                          end     = args.end)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RSNA Data Preprocessing Script")
    parser.add_argument('--dataset_path', required=True,   help='Path to the RSNA dataset')
    parser.add_argument('--output_path',  required=True,   help='Path to the RSNA dataset')
    parser.add_argument('--split',        default=None,    help='Dataset split (default: train)')
    parser.add_argument('--start',        default=0,       type=int,  help='Starting point Relative to dataset size')
    parser.add_argument('--end',          default=-1,      type=int,  help='Ending point Relative to dataset size')
    parser.add_argument('--limits', nargs='+',     type=int,              help='Optional limits as a list',default=None)

    args = parser.parse_args()
    main(args)
  