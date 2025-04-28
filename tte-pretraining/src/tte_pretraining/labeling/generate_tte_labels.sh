 #!/bin/bash

labeling_functions=('tte_mortality', 'tte_readmission', 'tte_PH', 'tte_Atelectasis' 'tte_Cardiomegaly' 'tte_Consolidation' 'tte_Edema' 'tte_Pleural_Effusion')

for labeling_function in "${labeling_functions[@]}"
do
python 'generate_tte_labels.py' \
--index_time_csv_path 'metadata_20250303.csv' \
--index_time_column 'procedure_DATETIME' \
--path_to_database 'femr_extract' \
--path_to_output_dir 'output' \
--labeling_function $labeling_function \
--is_skip_featurize \
--num_threads 12
done

