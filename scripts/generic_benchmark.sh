model=ot
eval_split=val
max_batch=1
batch_size_ip=4

#gaussian deblurring
problem=gaussian_deblurring_FFT
N=50
b_1=2.0
b_2=1.0
lmbda_0=0.01
pexp=1.0

dataset=celeba
method=generic_mmse_average
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} interpolation_mode zero steps_pnp ${N} max_batch ${max_batch} batch_size_ip ${batch_size_ip} b_1 ${b_1} b_2 ${b_2} lmbda_0 ${lmbda_0} pexp ${pexp}

# #method=pnp_flow
# #python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} interpolation_mode zero alpha 0.01 steps_pnp 100 max_batch ${max_batch} batch_size_ip ${batch_size_ip}

# dataset=afhq_cat
# method=generic_mmse_average
# python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} interpolation_mode zero steps_pnp ${N} max_batch ${max_batch} batch_size_ip ${batch_size_ip} b_1 ${b_1} b_2 ${b_2} lmbda_0 ${lmbda_0} pexp ${pexp}

#method=pnp_flow
#python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} interpolation_mode zero alpha 0.01 steps_pnp 500 max_batch ${max_batch} batch_size_ip ${batch_size_ip}


#superresolution
problem=superresolution
N=300
b_1=5.0
b_2=1.0
lmbda_0=0.01
pexp=1.0

dataset=celeba
method=generic_mmse_average
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} interpolation_mode zero steps_pnp ${N} max_batch ${max_batch} batch_size_ip ${batch_size_ip} b_1 ${b_1} b_2 ${b_2} lmbda_0 ${lmbda_0} pexp ${pexp}

#method=pnp_flow
#python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} interpolation_mode zero alpha 0.3 steps_pnp 100 max_batch ${max_batch} batch_size_ip ${batch_size_ip}

# dataset=afhq_cat
# method=generic_mmse_average
# python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} interpolation_mode zero steps_pnp ${N} max_batch ${max_batch} batch_size_ip ${batch_size_ip} b_1 ${b_1} b_2 ${b_2} lmbda_0 ${lmbda_0} pexp ${pexp}

#method=pnp_flow
#python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} interpolation_mode zero alpha 0.01 steps_pnp 500 max_batch ${max_batch} batch_size_ip ${batch_size_ip}


#random_inpainting
problem=random_inpainting
N=100
b_1=1.0
b_2=1.0
lmbda_0=0.01
pexp=1.0

dataset=celeba
method=generic_mmse_average
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} interpolation_mode zero steps_pnp ${N} max_batch ${max_batch} batch_size_ip ${batch_size_ip} b_1 ${b_1} b_2 ${b_2} lmbda_0 ${lmbda_0} pexp ${pexp}

# #method=pnp_flow
# #python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} interpolation_mode zero alpha 0.01 steps_pnp 100 max_batch ${max_batch} batch_size_ip ${batch_size_ip}

# dataset=afhq_cat
# method=generic_mmse_average
# python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} interpolation_mode zero steps_pnp ${N} max_batch ${max_batch} batch_size_ip ${batch_size_ip} b_1 ${b_1} b_2 ${b_2} lmbda_0 ${lmbda_0} pexp ${pexp}

# method=pnp_flow
# python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} interpolation_mode zero alpha 0.01 steps_pnp 200 max_batch ${max_batch} batch_size_ip ${batch_size_ip}


#inpainting
problem=inpainting
N=500
b_1=5.0
b_2=1.0
lmbda_0=0.01
pexp=1.0

dataset=celeba
method=generic_mmse_average
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} interpolation_mode zero steps_pnp ${N} max_batch ${max_batch} batch_size_ip ${batch_size_ip} b_1 ${b_1} b_2 ${b_2} lmbda_0 ${lmbda_0} pexp ${pexp}

# method=pnp_flow
# python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} interpolation_mode zero alpha 0.5 steps_pnp 100 max_batch ${max_batch} batch_size_ip ${batch_size_ip}

# dataset=afhq_cat
# method=generic_mmse_average
# python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} interpolation_mode zero steps_pnp ${N} max_batch ${max_batch} batch_size_ip ${batch_size_ip} b_1 ${b_1} b_2 ${b_2} lmbda_0 ${lmbda_0} pexp ${pexp}

# method=pnp_flow
# python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} interpolation_mode zero alpha 0.5 steps_pnp 100 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
