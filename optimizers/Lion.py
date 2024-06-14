import torch
from torch.optim.optimizer import Optimizer

class Lion(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), weight_decay=0, momentum=0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, momentum=momentum)
        super(Lion, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Lion does not support sparse gradients')
                
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['momentum_buffer'] = torch.zeros_like(p.data)

                exp_avg = state['exp_avg']
                buf = state['momentum_buffer']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Update exponential moving average of gradient values
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Apply weight decay if specified
                if group['weight_decay'] != 0:
                    grad.add_(p, alpha=group['weight_decay'])

                # Update momentum
                buf.mul_(group['momentum']).add_(exp_avg)

                # Update parameters
                p.data.add_(buf, alpha=-group['lr'])

        return loss
