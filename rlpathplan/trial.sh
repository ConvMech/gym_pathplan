source activate pathplan
export lr=0
export steps=5000000
export env=PathHallway-v0
#for expnum in 1 2 3 4 5 6 7 8 9 10
#for expnum in 11 12 13 14 15
#do
for p_index in 0
do
for seed in 100 101 102 
do
    python run_ppo.py --num-timesteps $steps --seed $seed --env $env --lr $lr --policy $p_index&
done
done
