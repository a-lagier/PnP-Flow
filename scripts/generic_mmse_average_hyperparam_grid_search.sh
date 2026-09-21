dataset=celeba
model=ot
eval_split=test
max_batch=1
batch_size_ip=4
# steps_pnp=100
lmbda_0=0.001
pexp=1.0

method=generic_mmse_average
for problem in superresolution random_inpainting #gaussian_deblurring_FFT
do
for N in 100 300 500
do
for b_1 in 0.5 1.0 2.0 5.0
do
for b_2 in 1.0 2.0 3.0
do
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} interpolation_mode zero lmbda_0 ${lmbda_0} pexp ${pexp} b_1 ${b_1} b_2 ${b_2} max_batch ${max_batch} batch_size_ip ${batch_size_ip} steps_pnp ${N}
done
done
done
done