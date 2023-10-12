# Copyright (c) 2023, HyBISCIS Team (Brown University, Boston University)
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 


import sys 

from torch.optim.lr_scheduler import StepLR, MultiStepLR, ConstantLR, \
    LinearLR, ExponentialLR, LambdaLR, CosineAnnealingLR


def get_scheduler(scheduler, optimizer, **kwargs):
    
    if scheduler == "StepLR": 
        step_size = kwargs.get("step_size", 4) # Period of learning rate decay
        gamma = kwargs.get("gamma", 0.5) # Multiplicative factor of learning rate decay 
        lr_scheduler = StepLR(optimizer, step_size = step_size, gamma = gamma) 

    elif scheduler == "MultiStepLR":
        milestones = kwargs.get("milestones", [8, 24, 28]) # List of epoch indices
        gamma = kwargs.get("gamma", 0.5) # Multiplicative factor of learning rate decay
        lr_scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma) 
    
    elif scheduler == "ConstantLR":
        factor = kwargs.get("factor", 0.5)  # The number we multiply learning rate until the milestone.
        total_iters = kwargs.get("total_iters", 30) # The number of steps that the scheduler decays the learning rate
        lr_scheduler = ConstantLR(optimizer, factor = factor, total_iters = total_iters) 
    
    elif scheduler == "LinearLR":
        factor = kwargs.get("factor", 0.5)
        start_factor = kwargs.get("start_factor", 0.5)  # The number we multiply learning rate in the first epoch
        total_iters = kwargs.get("start_factor", 8)  # The number of iterations that multiplicative factor reaches to 1 

        lr_scheduler = LinearLR(optimizer, start_factor = start_factor, total_iters = total_iters)     

    elif scheduler == "ExponentialLR":
        gamma = kwargs.get("gamma", 0.5) # Multiplicative factor of learning rate decay.
        lr_scheduler = ExponentialLR(optimizer, gamma = gamma) 
  
    elif scheduler == "LambdaLR":
        gamma = kwargs.get("gamma", 1)
        lr_lambda = lambda epoch: gamma ** epoch
        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    elif scheduler == "CosineLR":
        T = kwargs.get("T")
        lr_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=T, verbose=True)
    
    elif scheduler == "Poly":
        power = kwargs.get("power", 1)
        total_iters = kwargs.get("T", 1)
        # lr_scheduler = PolynomialLR(optimizer, total_iters=total_iters, power=power)
    else: 
        print(f"Invalid Name {scheduler} for learning rate scheduler.")
        sys.exit(1)


    return lr_scheduler

