import torch
import numpy as np
import os
from time import perf_counter
import pnpflow.image_generation.models.utils as mutils
import pnpflow.utils as utils



class OC_FLOW(object):

    def __init__(self, model, device, args):
        self.device = device
        self.args = args
        self.model = model.to(device)
        self.method = args.method


    def model_forward(self, x, t):
        if self.args.model == 'ot':
            return self.model(x, t)
        
        elif self.args.model == 'rectified':
            model_fn = mutils.get_model_fn(self.model, train=False)
            v = model_fn(x.type(torch.float), t * 999)
            return v

    def grad_datafit(self, x, y, H, H_adj):
        if self.args.noise_type == 'gaussian':
            return H_adj(H(x) - y) / (self.args.sigma_noise**2)
        elif self.args.noise_type == 'laplace':
            return H_adj(2*torch.heaviside(H(x)-y, torch.zeros_like(H(x)))-1)/self.args.sigma_noise
        else:
            raise ValueError('Noise type not supported')
    
    def solve_ip(self, test_loader, degradation, sigma_noise, H_funcs=None):
        torch.cuda.empty_cache()
        device = self.device
        self.args.sigma_noise = sigma_noise
        max_steps = self.args.max_steps
        ode_steps = self.args.ode_steps
        optim = self.args.optim
        max_optim_iter = self.args.max_optim_iter
        lr = self.args.lr
        H = degradation.H
        H_adj = degradation.H_adj

        loader = iter(test_loader)
        for batch in range(self.args.max_batch):
            (clean_img, labels) = next(loader)
            self.args.batch = batch


            if self.args.noise_type == 'gaussian':
                noisy_img = H(clean_img.clone().to(self.device))
                torch.manual_seed(batch)
                noisy_img += torch.randn_like(noisy_img) * sigma_noise
            elif self.args.noise_type == 'laplace':
                pass
        
            clean_img = clean_img.to('cpu')

            x_0 = torch.randn_like(clean_img).to(self.device)
            x = x_0
                
            timesteps = torch.linspace(0, 1, ode_steps + 1, device=device, dtype=torch.float)
            if optim == 'sgd':
                optimizer = torch.optim.SGD([du], lr=lr)
            else:
                optimizer = torch.optim.LBFGS([du], max_iter=max_optim_iter, lr=lr, line_search_fn='strong_wolfe')
                
            for step in range(max_steps):
                loss = optimizer.step(closure)
                print(f'Step {step}: Loss {loss:.4f}')

            du = du.detach()
            with torch.no_grad():
                x1_opt = odeint(
                    (lambda t, x: self.model_forward(x, t) + du[int(t * ode_steps)]),
                    x_0 + du[-1]
                ).detach()

            restored_img = x1_opt


            if self.args.save_results:
                restored_img = x.detach().clone()
                utils.save_images(clean_img, noisy_img, restored_img,
                                  self.args, H_adj, iter='final')
                utils.compute_psnr(clean_img, noisy_img,
                                   restored_img, self.args, H_adj, iter=max_steps)
                utils.compute_ssim(
                    clean_img, noisy_img, restored_img, self.args, H_adj, iter=max_steps)
                utils.compute_lpips(clean_img, noisy_img,
                                    restored_img, self.args, H_adj, iter=max_steps)

        if self.args.save_results:
            utils.compute_average_psnr(self.args)
            utils.compute_average_ssim(self.args)
            utils.compute_average_lpips(self.args)
        if self.args.compute_memory:
            utils.compute_average_memory(self.args)
        if self.args.compute_time:
            utils.compute_average_time(self.args)

    def should_save_image(self, iteration, steps):
        return iteration % (steps // 10) == 0

    def run_method(self, data_loaders, degradation, sigma_noise, H_funcs=None):

        # Construct the save path for results
        folder = utils.get_save_path_ip(self.args.dict_cfg_method)
        self.args.save_path_ip = os.path.join(self.args.save_path, folder)

        # Create the directory if it doesn't exist
        os.makedirs(self.args.save_path_ip, exist_ok=True)

        # Solve the inverse problem
        self.solve_ip(
            data_loaders[self.args.eval_split], degradation, sigma_noise, H_funcs)