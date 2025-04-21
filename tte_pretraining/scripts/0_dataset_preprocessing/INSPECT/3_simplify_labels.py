import argparse
from pe.datatasets.data_preprocessers.rsna_pe import  RSNAPePreprocessing
import os 

def main(args):
    #rsna_loader =  RSNAPePreprocessing(dataset_path=args.dataset_path)
    #rsna_loader.convert_studies_to_nifti(split=args.split,save_to=args.output_path,limits=args.limits)
    rsna_loader =  RSNAPePreprocessing(dataset_path="/share/pi/nigam/data/RSNAPE/")
    
    rsna_loader.simplfy_labels(split=args.split,save_to="/share/pi/nigam/data/RSNAPE/simplified_labels",limits=args.limits)
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RSNA Data Preprocessing Script")
    #parser.add_argument('--dataset_path', required=True,   help='Path to the RSNA dataset')
    #parser.add_argument('--output_path',  required=True,   help='Path to the RSNA dataset')
    parser.add_argument('--split',        default='train', help='Dataset split (default: train)')
    parser.add_argument('--limits', nargs='+', type=int, help='Optional limits as a list',default=None)

    args = parser.parse_args()
    main(args)