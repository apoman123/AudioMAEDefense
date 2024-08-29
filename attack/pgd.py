import torch
import torch.nn as nn

from attack import Attack


class PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ["default", "targeted"]

    def forward(self, input_data, labels, padding_masks=None, full_padding_masks=None):
        r"""
        Overridden.
        """
        input_list = []
        input_data = input_data.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(input_data, labels)

        loss = nn.CrossEntropyLoss()
        adv_input_data = input_data.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_input_data = adv_input_data + torch.empty_like(adv_input_data).uniform_(
                -self.eps, self.eps
            )
            adv_input_data = torch.clamp(adv_input_data, min=0, max=1).detach()

        # append to input list
        input_list.append(adv_input_data)

        if padding_masks != None:
            input_list.append(padding_masks)
        
        if full_padding_masks != None:
            input_list.append(full_padding_masks)


        for _ in range(self.steps):
            adv_input_data.requires_grad = True
            outputs = self.get_logits(*input_list)

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial input data
            grad = torch.autograd.grad(
                cost, adv_input_data, retain_graph=False, create_graph=False
            )[0]

            adv_input_data = adv_input_data.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_input_data - input_data, min=-self.eps, max=self.eps)
            adv_input_data = torch.clamp(input_data + delta, min=0, max=1).detach()

        return adv_input_data