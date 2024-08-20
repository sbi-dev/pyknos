# This file is part of pyknos, a library for conditional density estimation.
# pyknos is licensed under the Apache License 2.0., see LICENSE for more details.

from typing import Optional

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from pyknos.mdn.mdn import MultivariateGaussianMDN


def linear_gaussian(
    theta: Tensor, likelihood_shift: Tensor, likelihood_cov: Tensor
) -> Tensor:
    chol_factor = torch.cholesky(likelihood_cov)

    return likelihood_shift + theta + torch.mm(chol_factor, torch.randn_like(theta).T).T


def get_mdn(
    features: int,
    context_features: int,
    num_components: int = 10,
    hidden_features: Optional[int] = None,
) -> MultivariateGaussianMDN:
    if hidden_features is None:
        hidden_features = 50
    return MultivariateGaussianMDN(
        features=features,
        context_features=context_features,
        hidden_net=nn.Sequential(
            nn.Linear(context_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
        ),
        num_components=num_components,
        custom_initialization=True,
    )


@pytest.mark.parametrize("dim_input", ([1, 2]))
@pytest.mark.parametrize("dim_context", ([1, 2]))
@pytest.mark.parametrize(
    "device",
    (
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ),
)
@pytest.mark.parametrize("hidden_features", (10, None))
def test_mdn_for_diff_dimension_data(
    dim_input: int,
    dim_context: int,
    device: str,
    hidden_features: Optional[int],
    num_components: int = 2,
) -> None:
    num_samples = 5
    num_context = 1
    context = torch.randn(num_context, dim_context)

    net_features = hidden_features if hidden_features is not None else 20
    distribution = get_mdn(
        features=dim_input,
        context_features=dim_context,
        num_components=num_components,
        hidden_features=net_features,
    )
    distribution = distribution.to(device)

    # Test evaluation and sampling.
    samples = distribution.sample(num_samples, context=context.to(device))
    assert samples.shape == (num_context, num_samples, dim_input)

    log_probs = distribution.log_prob(
        samples.squeeze(0), context.to(device).repeat(num_samples, 1)
    )
    assert log_probs.shape == (num_samples,)
