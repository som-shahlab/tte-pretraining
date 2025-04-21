import argparse
from pe.datatasets.data_preprocessers.rsna_pe import  RSNAPePreprocessing
import os 

def main(args):
    rsna_loader =  RSNAPePreprocessing(dataset_path=args.dataset_path)
    rsna_loader.load_studies(split=args.split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RSNA Data Preprocessing Script")
    parser.add_argument('--dataset_path', required=True,   help='Path to the RSNA dataset')
    parser.add_argument('--split',        default='train', help='Dataset split (default: train)')

    args = parser.parse_args()
    main(args)