import torch

class SignSGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0, momentum=0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super(SignSGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.sign()  # Use the sign of the gradient

                state = self.state[p]

                # State initialization
                if 'momentum_buffer' not in state:
                    buf = state['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).add_(d_p)

                # Apply weight decay if specified
                if group['weight_decay'] != 0:
                    p.grad.add_(p, alpha=group['weight_decay'])

                p.add_(buf, alpha=-group['lr'])  # Update parameter

        return loss
