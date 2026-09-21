dataset=celeba
model=ot
eval_split=test
max_batch=1
batch_size_ip=1
lmbda_0=0.0
problem=denoising

for steps_pnp in 2000 3000
do
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method generic_mmse_average interpolation_mode zero lmbda_0 ${lmbda_0} b_1 0.0 b_2 1.0 max_batch ${max_batch} batch_size_ip ${batch_size_ip} steps_pnp ${steps_pnp}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method mmse_average interpolation_mode zero max_iter ${steps_pnp} max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method pnp_flow interpolation_mode zero max_batch ${max_batch} alpha 0.8 batch_size_ip ${batch_size_ip} steps_pnp ${steps_pnp}
done
