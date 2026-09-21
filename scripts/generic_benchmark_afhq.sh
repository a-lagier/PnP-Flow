model=ot
eval_split=test
max_batch=1
batch_size_ip=4
dataset=afhq
method=generic_mmse_average

gaussian deblurring
problem=gaussian_deblurring_FFT
N=100
b_1=15.0
b_2=2.0
lmbda_0=0.001
pexp=1.0

python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} interpolation_mode zero steps_pnp ${N} max_batch ${max_batch} batch_size_ip ${batch_size_ip} b_1 ${b_1} b_2 ${b_2} lmbda_0 ${lmbda_0} pexp ${pexp}

#superresolution
problem=superresolution
N=300
b_1=5.0
b_2=1.0
lmbda_0=0.01
pexp=1.0

python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} interpolation_mode zero steps_pnp ${N} max_batch ${max_batch} batch_size_ip ${batch_size_ip} b_1 ${b_1} b_2 ${b_2} lmbda_0 ${lmbda_0} pexp ${pexp}

#random_inpainting
problem=random_inpainting
N=100
b_1=1.0
b_2=1.0
lmbda_0=0.01
pexp=1.0

python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} interpolation_mode zero steps_pnp ${N} max_batch ${max_batch} batch_size_ip ${batch_size_ip} b_1 ${b_1} b_2 ${b_2} lmbda_0 ${lmbda_0} pexp ${pexp}

#inpainting
problem=inpainting
N=500
b_1=5.0
b_2=1.0
lmbda_0=0.01
pexp=1.0

python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} interpolation_mode zero steps_pnp ${N} max_batch ${max_batch} batch_size_ip ${batch_size_ip} b_1 ${b_1} b_2 ${b_2} lmbda_0 ${lmbda_0} pexp ${pexp}

#denoising
problem=denoising
N=50
b_1=0.0
b_2=0.0
lmbda_0=0.0
pexp=1.0

python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} interpolation_mode zero steps_pnp ${N} max_batch ${max_batch} batch_size_ip ${batch_size_ip} b_1 ${b_1} b_2 ${b_2} lmbda_0 ${lmbda_0} pexp ${pexp}