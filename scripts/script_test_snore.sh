dataset=afhq ## or celebahq or afhq_cat
model=ot  ## rectified for celebahq, gradient_step for method=pnp_gs (Hurault) or diffusion for method=pnp_diff (Zhu), ot otherwise.
eval_split=test
max_batch=5
batch_size_ip=4

# ### PNP FLOW
method=snore
# problem=denoising
# python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} max_iter 1500 gamma 0.1 sigma_0 0.5 sigma_last 1.8 alpha_0 0.025 alpha_last 3.24 max_batch ${max_batch} batch_size_ip ${batch_size_ip} steps_pnp 100
problem=gaussian_deblurring_FFT
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} max_iter 1500 gamma 0.1 sigma_0 1.8 sigma_last 0.5 alpha_0 0.025 alpha_last 3.24 max_batch ${max_batch} batch_size_ip ${batch_size_ip} 
# problem=superresolution
# python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} max_iter 1500 gamma 0.1 sigma_0 0.5 sigma_last 1.8 alpha_0 0.025 alpha_last 3.24 max_batch ${max_batch} batch_size_ip ${batch_size_ip} 
# problem=inpainting
# python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} max_iter 1500 gamma 0.1 sigma_0 0.5 sigma_last 1.8 alpha_0 0.025 alpha_last 3.24 max_batch ${max_batch} batch_size_ip ${batch_size_ip} 
problem=random_inpainting
# python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} max_iter 500 gamma 0.5 use_sigma_noise False max_batch ${max_batch} batch_size_ip ${batch_size_ip} steps_pnp 100




