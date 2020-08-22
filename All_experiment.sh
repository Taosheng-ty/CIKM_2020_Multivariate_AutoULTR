#!/bin/bash
# python run.py --json_file=example/Yahoo/DLCM_reverse.json
# python run.py --json_file=example/Yahoo/DLCM_init.json
# python run.py --json_file=example/Yahoo/DLCM_random.json
# python run.py --json_file=example/Yahoo/SetRank.json
# python run.py --json_file=example/Yahoo/DNN.json
# python run.py --json_file=example/Yahoo/DNN_naive.json
# python plot_results.py Yahoo

# python run.py --json_file=example/Istella-s/DLCM_reverse.json
# python run.py --json_file=example/Istella-s/DLCM_init.json
# python run.py --json_file=example/Istella-s/DLCM_random.json
# python run.py --json_file=example/Istella-s/SetRank.json
# python run.py --json_file=example/Istella-s/DNN.json
# python run.py --json_file=example/Istella-s/DNN_naive.json
# python plot_results.py Istella

python run.py --json_file=example/Toy/DLCM_reverse.json
python run.py --json_file=example/Toy/DLCM_init.json
python run.py --json_file=example/Toy/DLCM_random.json
python run.py --json_file=example/Toy/SetRank.json
python run.py --json_file=example/Toy/DNN.json
python run.py --json_file=example/Toy/DNN_naive.json
python plot_results.py Toy