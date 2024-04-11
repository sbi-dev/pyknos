from typing import Optional

import pytest
import torch
import torch.nn as nn
from torch import Tensor, eye

from pyknos.mdn.mdn import MultivariateGaussianMDN


def linear_gaussian(
    theta: Tensor, likelihood_shift: Tensor, likelihood_cov: Tensor
) -> Tensor:
    chol_factor = torch.cholesky(likelihood_cov)

    return likelihood_shift + theta + torch.mm(chol_factor, torch.randn_like(theta).T).T


@pytest.mark.parametrize("dim", ([1, 5, 10]))
@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
@pytest.mark.parametrize("hidden_features", (50, None))
def test_mdn_for_diff_dimension_data(
    dim: int, device: str, hidden_features: Optional[int], num_components: int = 10
) -> None:
    if device == "cuda:0" and not torch.cuda.is_available():
        pass
    else:
        theta = torch.rand(3, dim)
        likelihood_shift = torch.zeros(theta.shape)
        likelihood_cov = eye(dim)
        context = linear_gaussian(theta, likelihood_shift, likelihood_cov)

        x_numel = theta[0].numel()
        y_numel = context[0].numel()

        net_features = hidden_features if hidden_features is not None else 50
        distribution = MultivariateGaussianMDN(
            features=x_numel,
            context_features=y_numel,
            hidden_features=hidden_features,
            hidden_net=nn.Sequential(
                nn.Linear(y_numel, net_features),
                nn.ReLU(),
                nn.Linear(net_features, net_features),
                nn.ReLU(),
            ),
            num_components=num_components,
            custom_initialization=True,
        )
        distribution = distribution.to(device)

        logits, means, precisions, _, _ = distribution.get_mixture_components(
            theta.to(device)
        )

        # Test evaluation and sampling.
        distribution.log_prob(context.to(device), theta.to(device))
        distribution.sample(100, theta.to(device))
        distribution.sample_mog(10, logits, means, precisions)
