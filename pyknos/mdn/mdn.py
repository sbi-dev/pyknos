"""
Implementation of models based on
C. M. Bishop, "Mixture Density Networks", NCRG Report (1994)

Taken from http://github.com/conormdurkan/lfi. See there for copyright.
"""
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from nflows.utils import torchutils

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
        hidden_features: int,
        hidden_net: nn.Module,
        num_components: int,
        custom_initialization=False,
        embedding_net=None,
    ):
        """Mixture of multivariate Gaussians with full diagonal.

        Args:
            features: Dimension of output density.
            context_features: Dimension of inputs.
            hidden_features: Dimension of final layer of `hidden_net`.
            hidden_net: A Module which outputs final hidden representation before
                paramterization layers (i.e logits, means, and log precisions).
            num_components: Number of mixture components.
            custom_initialization: XXX
        """

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
        self.dimensions = None

        # XXX docstring text
        # embedding_net: NOT IMPLEMENTED
        #         A `nn.Module` which has trainable parameters to encode the
        #         context (conditioning). It is trained jointly with the MDN.
        if embedding_net is not None:
            raise NotImplementedError

        # Constant for numerical stability.
        self._epsilon = 1e-4

        # Initialize mixture coefficients and precision factors sensibly.
        if custom_initialization:
            self._initialize()

    def get_mixture_components(
        self, context: Tensor, dimensions: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return logits, means, precisions and two additional useful quantities.

        Args:
            context: Input to the MDN, leading dimension is batch dimension.

        Returns:
            A tuple with logits (num_components), means (num_components x output_dim),
            precisions (num_components, output_dim, output_dim), sum log diag of
            precision factors (1), precision factors (upper triangular precision factor
            A such that SIGMA^-1 = A^T A.) All batched.
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
        precisions[
            ..., torch.arange(self._features), torch.arange(self._features)
        ] += self._epsilon

        # For marginalization.
        diagonal = torch.permute(diagonal, (0, 2, 1))
        diagonal = torch.log(diagonal)
        diagonal[dimensions] = 0.0

        # The sum of the log diagonal of A is used in the likelihood calculation.
        sumlogdiag = torch.sum(diagonal, dim=1)

        return logits, means, precisions, sumlogdiag, precision_factors

    def log_prob(
        self,
        inputs: Tensor,
        context=Optional[Tensor],
    ) -> Tensor:
        """Return log MoG(inputs|context) where MoG is a mixture of Gaussians density.

        The MoG's parameters (mixture coefficients, means, and precisions) are the
        outputs of a neural network.

        Args:
            inputs: Input variable, leading dim interpreted as batch dimension.
            context: Conditioning variable, leading dim interpreted as batch dimension.
            dimensions: Must be a boolean tensor with the same shape as `inputs`.
                Only `True` elements are considered in the loss.

        Returns:
            Log probability of inputs given context under a MoG model.
        """

        # Obtain the dimensions that will be marginalized out.
        dimensions = torch.isnan(inputs)

        logits, means, precisions, sumlogdiag, _ = self.get_mixture_components(
            context, dimensions=dimensions
        )
        return self.log_prob_mog(
            inputs, logits, means, precisions, sumlogdiag, dimensions=dimensions
        )

    @staticmethod
    def log_prob_mog(
        inputs: Tensor,
        logits: Tensor,
        means: Tensor,
        precisions: Tensor,
        sumlogdiag: Tensor,
        dimensions: Tensor,
    ) -> Tensor:
        """
        Return the log-probability of `inputs` under a MoG with specified parameters.

        Unlike the `log_prob()` method, this method is fully detached from the neural
        network and can be used independent of the neural net in case the MoG
        parameters are already known.

        Args:
            inputs: Location at which to evaluate the MoG.
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

        # Split up evaluation into parts.
        a = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        b = -((output_dim - torch.sum(dimensions, dim=1)) / 2.0) * np.log(2 * np.pi)
        b = b.unsqueeze(1)
        c = sumlogdiag
        d1 = (inputs.expand_as(means) - means).view(
            batch_size, n_mixtures, output_dim, 1
        )

        d1 = torch.permute(d1, (0, 2, 1, 3))
        d1[dimensions] = 0.0
        d1 = torch.permute(d1, (0, 2, 1, 3))

        d2 = torch.matmul(precisions, d1)
        d2 = torch.permute(d2, (0, 2, 1, 3))
        d2[dimensions] = 0.0
        d2 = torch.permute(d2, (0, 2, 1, 3))

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
            context: Conditioning variable, leading dimension is batch dimension.

        Returns:
            Generated samples: (num_samples, output_dim) with leading batch dimension.
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
            Tensor: Samples from the MoG.
        """
        batch_size, n_mixtures, output_dim = means.shape

        # We need (batch_size * num_samples) samples in total.
        means, precision_factors = (
            torchutils.repeat_rows(means, num_samples),
            torchutils.repeat_rows(precision_factors, num_samples),
        )

        # Normalize the logits for the coefficients.
        coefficients = F.softmax(logits, dim=-1)  # [batch_size, num_components]

        # Choose num_samples mixture components per example in the batch.
        choices = torch.multinomial(
            coefficients, num_samples=num_samples, replacement=True
        ).view(
            -1
        )  # [batch_size, num_samples]

        # Create dummy index for indexing means and precision factors.
        ix = torchutils.repeat_rows(torch.arange(batch_size), num_samples)

        # Select means and precision factors.
        chosen_means = means[ix, choices, :]
        chosen_precision_factors = precision_factors[ix, choices, :, :]

        # Batch triangular solve to multiply standard normal samples by inverse
        # of upper triangular precision factor.
        zero_mean_samples, _ = torch.triangular_solve(
            torch.randn(
                batch_size * num_samples,
                output_dim,
                1,
                device=chosen_precision_factors.device,
            ),  # Need dummy final dimension.
            chosen_precision_factors,
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
        ) + self._epsilon * torch.randn(
            self._num_components * self._features
        )

        # Initialize off-diagonal of precision factors to zero.
        self._upper_layer.weight.data = self._epsilon * torch.randn(
            self._num_components * self._num_upper_params, self._hidden_features
        )
        self._upper_layer.bias.data = self._epsilon * torch.randn(
            self._num_components * self._num_upper_params
        )


class DiagnonalGaussValidClassifier(nn.Module):
    """
    A conditional density estimator that returns p(valid | theta) and p(x|theta) as a
    diagonal Gaussian.

    [1] Bishop, C.: 'Mixture Density Networks', Neural Computing Research Group Report
    1994 https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf
    """

    def __init__(
        self,
        features: int,
        context_features: int,
        hidden_features: int,
        hidden_net: nn.Module,
        custom_initialization=False,
        embedding_net=None,
        minimal_std=1.0,
    ):
        """Mixture of multivariate Gaussians with full diagonal.

        Args:
            features: Dimension of output density.
            context_features: Dimension of inputs.
            hidden_features: Dimension of final layer of `hidden_net`.
            hidden_net: A Module which outputs final hidden representation before
                paramterization layers (i.e logits, means, and log precisions).
            num_components: Number of mixture components.
            custom_initialization: XXX
        """

        super().__init__()

        self._features = features
        self._context_features = context_features
        self._hidden_features = hidden_features

        # Modules
        self._hidden_net = hidden_net

        self._means_layer = nn.Linear(hidden_features, features)
        self._unconstrained_diagonal_layer = nn.Linear(hidden_features, features)
        self._valid_layer = nn.Linear(hidden_features, features)

        # Constant for numerical stability.
        self._epsilon = 1e-4
        self._minimal_std = minimal_std

        # Initialize mixture coefficients and precision factors sensibly.
        if custom_initialization:
            self._initialize()

    def get_mixture_components(
        self, context: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return logits, means, precisions and two additional useful quantities.

        Args:
            context: Input to the MDN, leading dimension is batch dimension.

        Returns:
            A tuple with logits (num_components), means (num_components x output_dim),
            precisions (num_components, output_dim, output_dim), sum log diag of
            precision factors (1), precision factors (upper triangular precision factor
            A such that SIGMA^-1 = A^T A.) All batched.
        """

        h = self._hidden_net(context)
        means = self._means_layer(h)

        # Unconstrained diagonal and upper triangular quantities are unconstrained.
        unconstrained_diagonal = self._unconstrained_diagonal_layer(h)

        # Elements of diagonal of precision factor must be positive
        # (recall precision factor A such that SIGMA^-1 = A^T A).
        diagonal = (
            F.softplus(unconstrained_diagonal) + self._epsilon + self._minimal_std
        )

        valid_probs = F.sigmoid(self._valid_layer(h))

        return means, diagonal, valid_probs

    def log_prob(self, inputs: Tensor, context=Optional[Tensor]) -> Tensor:
        """Return log MoG(inputs|context) where MoG is a mixture of Gaussians density.

        The MoG's parameters (mixture coefficients, means, and precisions) are the
        outputs of a neural network.

        Args:
            inputs: Input variable, leading dim interpreted as batch dimension.
            context: Conditioning variable, leading dim interpreted as batch dimension.
            dimensions: Must be a boolean tensor with the same shape as `inputs`.
                Only `True` elements are considered in the loss.

        Returns:
            Log probability of inputs given context under a MoG model.
        """

        # Obtain the dimensions that will be marginalized out.
        protocol_dimensions = torch.isinf(inputs)
        dimensions = torch.isnan(inputs)
        dimensions = torch.logical_or(protocol_dimensions, dimensions)

        means, diagonal, predicted_valid_probs = self.get_mixture_components(context)
        mog_log_probs = self.log_prob_mog(
            inputs, means, diagonal, dimensions=dimensions
        )

        # Compute log-probs for classifier.
        valid_probs = torch.zeros(predicted_valid_probs.shape)
        valid_probs[~dimensions] = predicted_valid_probs[~dimensions]
        valid_probs[dimensions] = 1 - predicted_valid_probs[dimensions]
        valid_log_probs = torch.log(valid_probs)

        # Mask out those probs that were invalid because the protocol did not
        # get run. They should not influence the estimate of how often a particular
        # feature is valid. Simply setting this to 0 so it will not have a gradient.
        valid_log_probs[protocol_dimensions] = 0.0

        # For the invalid dimensions, the mog-log-prob does not play a role.
        # TODO: is this even necessary assuming that all invalid features are passed
        # as `NaN`?
        mog_log_probs[dimensions] = 0.0

        # Obtain the full log-prob given the gauss and the classifier.
        joint_probs = mog_log_probs + valid_log_probs

        return torch.sum(joint_probs, dim=1)

    @staticmethod
    def log_prob_mog(
        inputs: Tensor,
        means: Tensor,
        diagonal: Tensor,
        dimensions: Tensor,
    ) -> Tensor:
        """
        Return the log-probability of `inputs` under a MoG with specified parameters.

        Unlike the `log_prob()` method, this method is fully detached from the neural
        network and can be used independent of the neural net in case the MoG
        parameters are already known.

        Args:
            inputs: Location at which to evaluate the MoG.
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
        _, output_dim = means.size()

        # The sum of the log diagonal of A is used in the likelihood calculation.
        # For marginalization.
        diagonal_log = torch.log(diagonal)
        diagonal_log[dimensions] = 0.0
        sumlogdiag = diagonal_log

        # Split up evaluation into parts.
        b = -(1 / 2.0) * np.log(2 * np.pi)
        c = sumlogdiag

        d1 = inputs - means
        d1[dimensions] = 0.0
        d2 = diagonal ** 2 * d1
        d2[dimensions] = 0.0
        d = -0.5 * torch.einsum("bi, bi -> bi", d1, d2)

        return b + c + d

    def sample(self, num_samples: int, context: Tensor) -> Tensor:
        """
        Return num_samples independent samples from MoG(inputs | context).

        Generates num_samples samples for EACH item in context batch i.e. returns
        (num_samples * batch_size) samples in total.

        Args:
            num_samples: Number of samples to generate.
            context: Conditioning variable, leading dimension is batch dimension.

        Returns:
            Generated samples: (num_samples, output_dim) with leading batch dimension.
        """

        # Get necessary quantities.
        means, diagonal, _ = self.get_mixture_components(context)
        return self.sample_mog(num_samples, means, diagonal)

    @staticmethod
    def sample_mog(num_samples: int, means: Tensor, diagonal: Tensor) -> Tensor:
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
            Tensor: Samples from the MoG.
        """
        batch_size, output_dim = means.shape

        # We need (batch_size * num_samples) samples in total.
        means, diagonal = (
            torchutils.repeat_rows(means, num_samples),
            torchutils.repeat_rows(diagonal, num_samples),
        )

        # Batch triangular solve to multiply standard normal samples by inverse
        # of upper triangular precision factor.
        zero_mean_samples = (
            torch.randn((batch_size * num_samples, output_dim)) / diagonal
        )

        # Mow center samples at chosen means, removing dummy final dimension
        # from triangular solve.
        samples = means + zero_mean_samples

        return samples.reshape(batch_size, num_samples, output_dim)

    def _initialize(self) -> None:
        """
        Initialize MDN so that mixture coefficients are approximately uniform,
        and covariances are approximately the identity.
        """

        # Initialize diagonal of precision factors to inverse of softplus at 1.
        self._unconstrained_diagonal_layer.weight.data = self._epsilon * torch.randn(
            self._features, self._hidden_features
        )
        self._unconstrained_diagonal_layer.bias.data = torch.log(
            torch.exp(torch.tensor([1 - self._epsilon])) - 1
        ) * torch.ones(self._features) + self._epsilon * torch.randn(self._features)


# XXX This -> tests
def main():
    # probs = torch.Tensor([[1, 0], [0, 1]])
    # samples = torch.multinomial(probs, num_samples=5, replacement=True)
    # print(samples)
    # quit()
    mdn = MultivariateGaussianMDN(
        features=2,
        context_features=3,
        hidden_features=16,
        hidden_net=nn.Linear(3, 16),
        num_components=4,
    )
    inputs = torch.randn(1, 3)
    samples = mdn.sample(9, inputs)
    print(samples.shape)


if __name__ == "__main__":
    main()
