import torch
import numpy as np

def targeted_loss(inference_results, target_label, k):
    bsz , classes = inference_results.shape
    target_tensor = inference_results[:, target_label].unsqueeze(1).expand(bsz, classes)
    logit_differences = inference_results - target_tensor
    max_logit_differences = torch.max(logit_differences, dim=-1, keepdim=True)[0]
    losses = torch.max(torch.cat([max_logit_differences, torch.tensor([-k for one_data in range(bsz)]).unsqueeze(1)], dim=1), dim=-1)
    return losses

def untargeted_loss(inference_results, target_label, k):
    bsz, classes = inference_results.shape
    label_logits = torch.tensor([])
    for idx in range(bsz):
        label_logits = torch.cat([label_logits, inference_results[idx][target_label[idx]]], dim=0)
        inference_results[idx][target_label[idx]] = -torch.inf
    
    max_class_logits = torch.max(inference_results, dim=-1, keepdim=True)[0]
    difference = label_logits - max_class_logits
    ks = torch.tensor([-k for _ in range(bsz)])
    losses = torch.max(torch.cat([difference, ks], dim=-1), dim=-1)

    return losses


class FakeBob():
    def __init__(self, model, lr, steps=200, epsilon=0.002, norm_type=np.inf, targeted=False, sigma=1e-3, m=50, k=0, momemtum=0.9) -> None:
        self.model = model
        self.lr = lr
        self.sigma = sigma
        self.m = m
        self.k = k
        self.momemtum = momemtum
        self.steps = steps
        self.epsilon = epsilon
        self.targeted = targeted

    def calc_grad(self, input_data, target_label, padding_masks=None, full_padding_masks=None, ):
        bsz = input_data.shape[0]

        # noises
        noises = [torch.randn(input_data.shape) for _ in range(self.m)]
        noisy_input_data = [input_data + noise * self.sigma for noise in noises] 

        # inference results
        results = [self.model(one_noisy_input_data) for one_noisy_input_data in noisy_input_data]
        if self.targeted:
            losses = torch.tensor([targeted_loss(result) for result in results])
        else:
            losses = torch.tensor([untargeted_loss(result) for result in results])
        
        noises = torch.tensor(noises)
        
        return (losses * noises) / (self.m * self.sigma)

    def forward(self, input_data, target_label, padding_masks=None, full_padding_masks=None):

        lower = torch.clip(input_data - self.epsilon, min=-1)
        higher = torch.clip(input_data + self.epsilon, max=1)

        # bim steps
        pregrad = None
        for step in range(self.steps):
            grad = self.calc_grad(input_data, target_label, padding_masks, full_padding_masks)

            if pregrad != None:
                grad = self.momemtum * pregrad + (1.0 - self.momemtum) * grad

            pregrad = grad

            # update the adversarial examples
            input_data -= self.lr * grad.sign()
            input_data = torch.clamp(input_data, lower, higher)

        return input_data


            
            

        
            