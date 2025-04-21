import datasets
import pickle
import argparse
from pathlib import Path



"""
python 1a_prune_ontology.py \
--input-dataset "/share/pi/nigam/projects/zphuo/data/PE/inspect/timelines_smallfiles_meds/data/*" \
--input-ontology "/share/pi/nigam/projects/zphuo/data/PE/inspect/ontology.pkl" \
--output-ontology "/share/pi/nigam/projects/zphuo/data/PE/inspect/inspect_ontology.pkl" \
--num-processors 32 
"""



def process_ontology(
    input_dataset_path: str,
    input_ontology_path: str,
    output_ontology_path: str,
    num_processors: int = 32,
    ontologies_to_remove: set = {'SPL', 'ATC', 'HemOnc', 'LOINC'}
):
    """
    Process and prune ontology based on the given dataset.
    
    Args:
        input_dataset_path: Path to the parquet dataset
        input_ontology_path: Path to input ontology pickle file
        output_ontology_path: Path to save processed ontology
        num_processors: Number of processors to use
        ontologies_to_remove: Set of ontologies to remove
    """
    # Load dataset
    dataset = datasets.Dataset.from_parquet(input_dataset_path)
    
    # Load ontology
    with open(input_ontology_path, 'rb') as f:
        ontology = pickle.load(f)
    
    # Prune ontology
    ontology.prune_to_dataset(
        dataset,
        num_proc=num_processors,
        prune_all_descriptions=True,
        remove_ontologies=ontologies_to_remove
    )
    
    # Save processed ontology
    output_path = Path(output_ontology_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(ontology, f)

def main():
    parser = argparse.ArgumentParser(description='Process and prune ontology based on dataset')
    parser.add_argument('--input-dataset', type=str, required=True,
                        help='Path to input parquet dataset')
    parser.add_argument('--input-ontology', type=str, required=True,
                        help='Path to input ontology pickle file')
    parser.add_argument('--output-ontology', type=str, required=True,
                        help='Path to save processed ontology')
    parser.add_argument('--num-processors', type=int, default=32,
                        help='Number of processors to use')
    parser.add_argument('--remove-ontologies', type=str, nargs='+',
                        default=['SPL', 'ATC', 'HemOnc', 'LOINC'],
                        help='List of ontologies to remove')
    
    args = parser.parse_args()
    
    process_ontology(
        args.input_dataset,
        args.input_ontology,
        args.output_ontology,
        args.num_processors,
        set(args.remove_ontologies)
    )

if __name__ == '__main__':
    main()