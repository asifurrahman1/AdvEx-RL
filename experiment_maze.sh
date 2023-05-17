#!/bin/bash

"Experiments on Maze Environment"
python test_aaa_atk_experiment.py --configure-env maze --exp-data-dir '/Experimental_Data/Reproduced' 
python test_random_atk_experiment.py --configure-env maze --exp-data-dir '/Experimental_Data/Reproduced'

echo "Changed Environment dynamics $i"
for i in {2..10..2}
do
	python test_aaa_atk_experiment_changed_dynamics.py --configure-env maze --env-change $i --exp-data-dir '/Experimental_Data/Reproduced/Changed_dynamics'
done
python test_ablation_random_atk.py --configure-env maze --exp-data-dir '/Experimental_Data/Reproduced/Ablation' 
python test_ablation_aaa_atk.py --configure-env maze --exp-data-dir '/Experimental_Data/Reproduced/Ablation'
