#!/bin/bash
set -e
cd /data/home/aviyac/retailers_with_shipping/N50/rho_sweep
source /data/home/aviyac/ap_env/bin/activate

python3 run_rho_seed.py --rho 2.000 --seed 8 --max_iter 25 --force &
python3 run_rho_seed.py --rho 2.000 --seed 2 --max_iter 25 --force &
python3 run_rho_seed.py --rho 2.000 --seed 6 --max_iter 25 --force &
python3 run_rho_seed.py --rho 2.000 --seed 4 --max_iter 25 --force &
python3 run_rho_seed.py --rho 2.000 --seed 7 --max_iter 25 --force &
python3 run_rho_seed.py --rho 2.000 --seed 9 --max_iter 25 --force &
python3 run_rho_seed.py --rho 2.000 --seed 5 --max_iter 25 --force &
python3 run_rho_seed.py --rho 1.000 --seed 9 --max_iter 25 --force &
python3 run_rho_seed.py --rho 1.000 --seed 0 --max_iter 25 --force &
python3 run_rho_seed.py --rho 0.400 --seed 4 --max_iter 25 --force &
python3 run_rho_seed.py --rho 0.300 --seed 3 --max_iter 25 --force &
python3 run_rho_seed.py --rho 0.300 --seed 9 --max_iter 25 --force &
python3 run_rho_seed.py --rho 1.500 --seed 7 --max_iter 25 --force &
python3 run_rho_seed.py --rho 1.500 --seed 9 --max_iter 25 --force &
python3 run_rho_seed.py --rho 0.700 --seed 0 --max_iter 25 --force &
python3 run_rho_seed.py --rho 0.500 --seed 9 --max_iter 25 --force &
python3 run_rho_seed.py --rho 0.500 --seed 5 --max_iter 25 --force &
wait
echo done
