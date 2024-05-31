import torch
from torch.optim import Optimizer

class Adahessian(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-4, hessian_power=0.5):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, hessian_power=hessian_power)
        super(Adahessian, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_hessian_diag_sq'] = torch.zeros_like(p)

                exp_avg, exp_hessian_diag_sq = state['exp_avg'], state['exp_hessian_diag_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                hessian_diag_sq = self.compute_hessian_diag_sq(grad, p)

                exp_hessian_diag_sq.mul_(beta2).add_(hessian_diag_sq, alpha=1 - beta2)

                denom = (exp_hessian_diag_sq.sqrt().add_(group['eps']) ** group['hessian_power']).reciprocal()
                step_size = group['lr'] * denom

                p.addcdiv_(exp_avg, step_size, value=-1)

        return loss

    def compute_hessian_diag_sq(self, grad, p):
        hessian_diag_sq = torch.zeros_like(p)
        grad_squared = grad.pow(2)

        for i in range(len(grad)):
            grad[i].backward(retain_graph=True)
            hessian_diag_sq[i] = p.grad.data.clone()
            p.grad.data.zero_()

        return hessian_diag_sq