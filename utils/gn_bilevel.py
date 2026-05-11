import torch


class Architect(object):
    def __init__(self, model, args):
        self.model = model

        self.optimizer = torch.optim.AdamW(
            self.model.arch_parameters(),
            lr=getattr(args, "arch_learning_rate", 1e-3),
        )

    def step(self, lamda, latency, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled=True):
        self.optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(lamda, latency, input_train, target_train, input_valid, target_valid, eta, network_optimizer)
        else:
            self._backward_step(input_valid, target_valid, lamda, latency)
        self.optimizer.step()

    def _backward_step(self, input_valid, target_valid, lamda, latency):
        loss = self.model._loss(input_valid, target_valid, lamda, latency)
        loss.backward()

    def _backward_step_unrolled(self, lamda, latency, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
        unrolled_loss = self.model._loss(input_valid, target_valid, lamda, latency)
        unrolled_loss.backward()

        dalpha = []
        for v in self.model.arch_parameters():
            if v.grad is None:
                dalpha.append(torch.zeros_like(v))
            else:
                dalpha.append(v.grad.detach().clone())

        vector = []
        for v in self.model.parameters():
            if v.grad is None:
                vector.append(torch.zeros_like(v))
            else:
                vector.append(v.grad.detach().clone())

        lower_loss = self.model._loss(input_train, target_train, lamda, latency)
        dfy = torch.autograd.grad(lower_loss, self.model.parameters(), allow_unused=True)

        gfyfy = 0
        gFyfy = 0
        for f, F_vec in zip(dfy, vector):
            if f is None:
                f = torch.zeros_like(F_vec)
            gfyfy = gfyfy + torch.sum(f * f)
            gFyfy = gFyfy + torch.sum(F_vec * f)

        lower_loss_2 = self.model._loss(input_train, target_train, lamda, latency)
        gn_loss = -gFyfy.detach() / (gfyfy.detach() + 1e-12) * lower_loss_2
        implicit_grads = torch.autograd.grad(gn_loss, self.model.arch_parameters(), allow_unused=True)

        new_dalpha = []
        for g, ig, v in zip(dalpha, implicit_grads, self.model.arch_parameters()):
            if g is None and ig is None:
                new_g = torch.zeros_like(v)
            else:
                if g is None:
                    g = torch.zeros_like(v)
                if ig is None:
                    ig = torch.zeros_like(v)
                new_g = g - eta * ig
            new_dalpha.append(new_g)

        for v, g in zip(self.model.arch_parameters(), new_dalpha):
            if v.grad is None:
                v.grad = g.detach().clone()
            else:
                v.grad.data.copy_(g.data)
