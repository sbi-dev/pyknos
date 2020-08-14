import pytest
import torch
from torch import Tensor, eye, ones, zeros
import torch.nn as nn
from torch.distributions import MultivariateNormal

from pyknos.mdn.mdn import MultivariateGaussianMDN


def linear_gaussian(
    theta: Tensor,
    likelihood_shift: Tensor,
    likelihood_cov: Tensor,
) -> Tensor:

    chol_factor = torch.cholesky(likelihood_cov)

    return likelihood_shift + theta + torch.mm(chol_factor, torch.randn_like(theta).T).T


@pytest.mark.parametrize("dim", ([1, 5, 10]),)
def test_mdn_for_diff_dimension_data(dim: int, 
            hidden_features: int=50, 
            num_components: int=10) -> None:

    theta = torch.rand(3, dim)
    likelihood_shift = torch.zeros(theta.shape)
    likelihood_cov = eye(dim)
    context =  linear_gaussian(theta, likelihood_shift, likelihood_cov)

    x_numel = theta[0].numel()
    y_numel = context[0].numel()
    
    distribution = MultivariateGaussianMDN(
        features=x_numel,
        context_features=y_numel,
        hidden_features=hidden_features,
        hidden_net=nn.Sequential(
            nn.Linear(y_numel, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
        ),
        num_components=num_components,
        custom_initialization=True,
    )

    logits, means, precisions, _, _ = distribution.get_mixture_components(theta)
    
    log_prob = distribution.log_prob(context, theta)

    sample = distribution.sample(100, theta)

