"""
Implementation of models based on
C. M. Bishop, "Mixture Density Networks", NCRG Report (1994)

Taken from http://github.com/conormdurkan/lfi. See there for copyright.
"""

import warnings
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


# This implementation based on Conor M. Durkan's et al. lfi package (2020).
# https://github.com/conormdurkan/lfi/blob/master/src/nn_/nde/mdn.py
class MultivariateGaussianMDN(nn.Module):
    """
    Conditional density mixture of multivariate Gaussians, after Bishop [1].

    A multivariate Gaussian mixture with full (rather than diagonal) covariances

    [1] Bishop, C.: 'Mixture Density Networks', Neural Computing Research Group Report
    1994 https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf
    """

    def __init__(
        self,
        features: int,
        context_features: int,
        hidden_net: Optional[nn.Module],
        num_components: int,
        hidden_features: Optional[int] = None,
        custom_initialization: bool = False,
        embedding_net: Optional[nn.Module] = None,
    ):
        """Mixture of multivariate Gaussians with full diagonal.

        Args:
            features: Dimension of output density.
            context_features: Dimension of inputs.
            hidden_features: Dimension of final layer of `hidden_net`.
            hidden_net: A Module which outputs final hidden representation before
                paramterization layers (i.e logits, means, and log precisions).
            num_components: Number of mixture components.
            custom_initialization: If True, initialize mixture coefficients to be
                approximately uniform and covariances to be approximately the identity.
        """

        # Infer hidden_features from hidden_net if not provided.
        if hidden_net is None:
            hidden_net = nn.Identity()
        try:
            inferred_hidden_features: int = hidden_net(
                torch.randn(1, context_features)
            ).shape[-1]
        except IndexError as err:
            raise IndexError(
                "Could not infer hidden_features from hidden_net. Please provide "
                "hidden_features explicitly."
            ) from err

        if hidden_features is not None:
            warnings.warn(
                "'hidden_features' parameter is deprecated and will be removed "
                "in a future version. Pass a hidden_net instead and the resulting "
                "hidden_features will be inferred from it.",
                DeprecationWarning,
                stacklevel=2,
            )
            assert hidden_features == inferred_hidden_features, (
                f"hidden_features={hidden_features} does not match inferred value "
                f"{inferred_hidden_features} from hidden_net."
            )
        else:
            hidden_features = inferred_hidden_features

        super().__init__()

        self._features = features
        self._context_features = context_features
        self._hidden_features = hidden_features
        self._num_components = num_components

        self._num_upper_params = (features * (features - 1)) // 2

        self._row_ix, self._column_ix = np.triu_indices(features, k=1)
        self._diag_ix = range(features)

        # Modules
        self._hidden_net = hidden_net

        self._logits_layer = nn.Linear(hidden_features, num_components)

        self._means_layer = nn.Linear(hidden_features, num_components * features)

        self._unconstrained_diagonal_layer = nn.Linear(
            hidden_features, num_components * features
        )
        self._upper_layer = nn.Linear(
            hidden_features, num_components * self._num_upper_params
        )

        if embedding_net is not None:
            raise NotImplementedError(
                "embedding net is not implemented yet. We recommend using MDN in ",
                "conjunction with nflows as done in the sbi package.",
            )

        # Constant for numerical stability.
        self._epsilon = 1e-4

        # Initialize mixture coefficients and precision factors sensibly.
        if custom_initialization:
            self._initialize()

    def get_mixture_components(
        self, context: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return logits, means, precisions and two additional useful quantities.

        Args:
            context: Input to the MDN, leading dimension is batch
                (batch_size, context_features).

        Returns:
            A tuple with
                logits (batch, num_components)
                means (batch, num_components, output_dim),
                precisions (batch, num_components, output_dim, output_dim),
                sum log diag of precision factors (batch, 1),
                precision factors (upper triangular precision factor A such that
                    SIGMA^-1 = A^T A.) (batch, num_components, output_dim, output_dim)
        """

        h = self._hidden_net(context)

        # Logits and Means are unconstrained and are obtained directly from the
        # output of a linear layer.
        logits = self._logits_layer(h)
        means = self._means_layer(h).view(-1, self._num_components, self._features)

        # Unconstrained diagonal and upper triangular quantities are unconstrained.
        unconstrained_diagonal = self._unconstrained_diagonal_layer(h).view(
            -1, self._num_components, self._features
        )

        # Elements of diagonal of precision factor must be positive
        # (recall precision factor A such that SIGMA^-1 = A^T A).
        diagonal = F.softplus(unconstrained_diagonal)

        # Create empty precision factor matrix, and fill with appropriate quantities.
        precision_factors = torch.zeros(
            means.shape[0],
            self._num_components,
            self._features,
            self._features,
            device=context.device,
        )
        precision_factors[..., self._diag_ix, self._diag_ix] = diagonal

        # one dimensional feature does not involve upper triangular parameters
        if self._features > 1:
            upper = self._upper_layer(h).view(
                -1, self._num_components, self._num_upper_params
            )
            precision_factors[..., self._row_ix, self._column_ix] = upper

        # Precisions are given by SIGMA^-1 = A^T A.
        precisions = torch.matmul(
            torch.transpose(precision_factors, 2, 3), precision_factors
        )
        # Add epsilon to diagnonal for numerical stability.
        precisions[..., torch.arange(self._features), torch.arange(self._features)] += (
            self._epsilon
        )

        # The sum of the log diagonal of A is used in the likelihood calculation.
        sumlogdiag = torch.sum(torch.log(diagonal), dim=-1)

        return logits, means, precisions, sumlogdiag, precision_factors

    def log_prob(self, inputs: Tensor, context: Tensor) -> Tensor:
        """Return log MoG(inputs|context) where MoG is a mixture of Gaussians density.

        The MoG's parameters (mixture coefficients, means, and precisions) are the
        outputs of a neural network.

        Args:
            inputs: Input variable, (batch, dim_input).
            context: Conditioning variable, (batch, dim_context).

        Returns:
            Log probability of inputs given context under a MoG model, (batch).
        """

        logits, means, precisions, sumlogdiag, _ = self.get_mixture_components(context)
        return self.log_prob_mog(inputs, logits, means, precisions, sumlogdiag)

    @staticmethod
    def log_prob_mog(
        inputs: Tensor,
        logits: Tensor,
        means: Tensor,
        precisions: Tensor,
        sumlogdiag: Tensor,
    ) -> Tensor:
        """
        Return the log-probability of `inputs` under a MoG with specified parameters.

        Unlike the `log_prob()` method, this method is fully detached from the neural
        network and can be used independent of the neural net in case the MoG
        parameters are already known.

        Args:
            inputs: Location at which to evaluate the MoG (batch_size, dim_input).
            logits: Log-weights of each component of the MoG. Shape: (batch_size,
                num_components).
            means: Means of each MoG, shape (batch_size, num_components, parameter_dim).
            precisions: Precision matrices of each MoG. Shape:
                (batch_size, num_components, parameter_dim, parameter_dim).
            sumlogdiag: Sum of the logarithm of the diagonal of the precision factors.
                Shape: (batch_size, num_components).

        Returns:
            Log-probabilities of each input.
        """
        batch_size, n_mixtures, output_dim = means.size()
        inputs = inputs.view(-1, 1, output_dim)
        assert inputs.size(0) == batch_size, "Batch size of inputs does not match."

        # Split up evaluation into parts.
        a = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        b = -(output_dim / 2.0) * np.log(2 * np.pi)
        c = sumlogdiag
        d1 = (inputs.expand_as(means) - means).view(
            batch_size, n_mixtures, output_dim, 1
        )
        d2 = torch.matmul(precisions, d1)
        d = -0.5 * torch.matmul(torch.transpose(d1, 2, 3), d2).view(
            batch_size, n_mixtures
        )

        return torch.logsumexp(a + b + c + d, dim=-1)

    def sample(self, num_samples: int, context: Tensor) -> Tensor:
        """
        Return num_samples independent samples from MoG(inputs | context).

        Generates num_samples samples for EACH item in context batch i.e. returns
        (num_samples * batch_size) samples in total.

        Args:
            num_samples: Number of samples to generate.
            context: Conditioning variable, (batch_size, context_dim).

        Returns:
            Generated samples: (batch_size, num_samples, output_dim) with leading batch
            dimension.
        """

        # Get necessary quantities.
        logits, means, _, _, precision_factors = self.get_mixture_components(context)
        return self.sample_mog(num_samples, logits, means, precision_factors)

    @staticmethod
    def sample_mog(
        num_samples: int, logits: Tensor, means: Tensor, precision_factors: Tensor
    ) -> Tensor:
        """
        Return samples of a MoG with specified parameters.

        Unlike the `sample()` method, this method is fully detached from the neural
        network and can be used independent of the neural net in case the MoG
        parameters are already known.

        Args:
            num_samples: Number of samples to generate.
            logits: Log-weights of each component of the MoG. Shape: (batch_size,
                num_components).
            means: Means of each MoG. Shape: (batch_size, num_components,
                parameter_dim).
            precision_factors: Cholesky factors of each component of the MoG. Shape:
                (batch_size, num_components, parameter_dim, parameter_dim).

        Returns:
            Tensor: Samples from the MoG (batch_size, num_samples, output_dim).
        """
        batch_size, _, output_dim = means.shape

        # Normalize the logits for the coefficients.
        coefficients = F.softmax(logits, dim=-1)  # [batch_size, num_components]

        # Choose num_samples mixture components per example in the batch.
        choices = torch.multinomial(
            coefficients, num_samples=num_samples, replacement=True
        )  # [batch_size, num_samples]

        # Create dummy index for indexing means and precision factors.
        ix = torch.arange(batch_size).unsqueeze(1).expand(batch_size, num_samples)

        # Select means and precision factors.
        chosen_means = means[ix, choices, :]
        chosen_precision_factors = precision_factors[ix, choices, :, :]

        # Batch triangular solve to multiply standard normal samples by inverse
        # of upper triangular precision factor.
        zero_mean_samples = torch.linalg.solve_triangular(
            chosen_precision_factors,
            torch.randn(
                batch_size,
                num_samples,
                output_dim,
                1,
                device=chosen_precision_factors.device,
            ),  # Need dummy final dimension.
            upper=True,
        )

        # Mow center samples at chosen means, removing dummy final dimension
        # from triangular solve.
        samples = chosen_means + zero_mean_samples.squeeze(-1)

        return samples.reshape(batch_size, num_samples, output_dim)

    def _initialize(self) -> None:
        """
        Initialize MDN so that mixture coefficients are approximately uniform,
        and covariances are approximately the identity.
        """

        # Initialize mixture coefficients to near uniform.
        self._logits_layer.weight.data = self._epsilon * torch.randn(
            self._num_components, self._hidden_features
        )
        self._logits_layer.bias.data = self._epsilon * torch.randn(self._num_components)

        # Initialize diagonal of precision factors to inverse of softplus at 1.
        self._unconstrained_diagonal_layer.weight.data = self._epsilon * torch.randn(
            self._num_components * self._features, self._hidden_features
        )
        self._unconstrained_diagonal_layer.bias.data = torch.log(
            torch.exp(torch.tensor([1 - self._epsilon])) - 1
        ) * torch.ones(
            self._num_components * self._features
        ) + self._epsilon * torch.randn(self._num_components * self._features)

        # Initialize off-diagonal of precision factors to zero.
        self._upper_layer.weight.data = self._epsilon * torch.randn(
            self._num_components * self._num_upper_params, self._hidden_features
        )
        self._upper_layer.bias.data = self._epsilon * torch.randn(
            self._num_components * self._num_upper_params
        )
