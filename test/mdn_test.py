import pytest
import torch
import torch.nn as nn
from torch import Tensor, eye, ones, zeros
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

    y = torch.rand(3, dim)
    likelihood_shift = torch.zeros(y.shape)
    likelihood_cov = eye(dim)
    x =  linear_gaussian(y, likelihood_shift, likelihood_cov)

    x_numel = x[0].numel()
    
    distribution = MultivariateGaussianMDN(
        features=dim,
        context_features=x_numel,
        hidden_features=hidden_features,
        hidden_net=nn.Sequential(
            nn.Linear(x_numel, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
        ),
        num_components=num_components,
        custom_initialization=True,
    )

    logits, means, precisions, _, _ = distribution.get_mixture_components(y)
    
    log_prob = distribution.log_prob(x, y)

    sample = distribution.sample(100, y)

