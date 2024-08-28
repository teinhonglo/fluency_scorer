ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)
LOG2PI = math.log(2 * math.pi)

class MixtureDensityHeadConfig:
    """MixtureDensityHead configuration.

    Args:
        num_gaussian (int): Number of Gaussian Distributions in the mixture model. Defaults to 1

        sigma_bias_flag (bool): Whether to have a bias term in the sigma layer. Defaults to False

        mu_bias_init (Optional[List]): To initialize the bias parameter of the mu layer to predefined
                cluster centers. Should be a list with the same length as number of gaussians in the mixture
                model. It is highly recommended to set the parameter to combat mode collapse. Defaults to None

        weight_regularization (Optional[int]): Whether to apply L1 or L2 Norm to the MDN layers. Defaults
                to L2. Choices are: [`1`,`2`].

        lambda_sigma (Optional[float]): The regularization constant for weight regularization of sigma
                layer. Defaults to 0.1

        lambda_pi (Optional[float]): The regularization constant for weight regularization of pi layer.
                Defaults to 0.1

        lambda_mu (Optional[float]): The regularization constant for weight regularization of mu layer.
                Defaults to 0

        softmax_temperature (Optional[float]): The temperature to be used in the gumbel softmax of the
                mixing coefficients. Values less than one leads to sharper transition between the multiple
                components. Defaults to 1

        n_samples (int): Number of samples to draw from the posterior to get prediction. Defaults to 100

        central_tendency (str): Which measure to use to get the point prediction. Defaults to mean. Choices
                are: [`mean`,`median`].

        speedup_training (bool): Turning on this parameter does away with sampling during training which
                speeds up training, but also doesn't give you visibility on train metrics. Defaults to False

        log_debug_plot (bool): Turning on this parameter plots histograms of the mu, sigma, and pi layers
                in addition to the logits(if log_logits is turned on in experment config). Defaults to False

        input_dim (int): The input dimensions to the head. This will be automatically filled in while
                initializing from the `backbone.output_dim`

    """

    num_gaussian: int = field(
        default=1,
        metadata={
            "help": "Number of Gaussian Distributions in the mixture model. Defaults to 1",
        },
    )
    sigma_bias_flag: bool = field(
        default=False,
        metadata={
            "help": "Whether to have a bias term in the sigma layer. Defaults to False",
        },
    )
    mu_bias_init: Optional[List] = field(
        default=None,
        metadata={
            "help": "To initialize the bias parameter of the mu layer to predefined cluster centers."
            " Should be a list with the same length as number of gaussians in the mixture model."
            " It is highly recommended to set the parameter to combat mode collapse. Defaults to None",
        },
    )

    weight_regularization: Optional[int] = field(
        default=2,
        metadata={
            "help": "Whether to apply L1 or L2 Norm to the MDN layers. Defaults to L2",
            "choices": [1, 2],
        },
    )

    lambda_sigma: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "The regularization constant for weight regularization of sigma layer. Defaults to 0.1",
        },
    )
    lambda_pi: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "The regularization constant for weight regularization of pi layer. Defaults to 0.1",
        },
    )
    lambda_mu: Optional[float] = field(
        default=0,
        metadata={
            "help": "The regularization constant for weight regularization of mu layer. Defaults to 0",
        },
    )
    softmax_temperature: Optional[float] = field(
        default=1,
        metadata={
            "help": "The temperature to be used in the gumbel softmax of the mixing coefficients."
            " Values less than one leads to sharper transition between the multiple components. Defaults to 1",
        },
    )
    n_samples: int = field(
        default=100,
        metadata={
            "help": "Number of samples to draw from the posterior to get prediction. Defaults to 100",
        },
    )
    central_tendency: str = field(
        default="mean",
        metadata={
            "help": "Which measure to use to get the point prediction. Defaults to mean",
            "choices": ["mean", "median"],
        },
    )
    speedup_training: bool = field(
        default=False,
        metadata={
            "help": "Turning on this parameter does away with sampling during training which speeds up training,"
            " but also doesn't give you visibility on train metrics. Defaults to False",
        },
    )
    log_debug_plot: bool = field(
        default=False,
        metadata={
            "help": "Turning on this parameter plots histograms of the mu, sigma, and pi layers in addition"
            " to the logits(if log_logits is turned on in experment config). Defaults to False",
        },
    )
    input_dim: int = field(
        default=None,
        metadata={
            "help": "The input dimensions to the head. This will be automatically filled in while initializing"
            " from the `backbone.output_dim`",
        },
    )
    _probabilistic: bool = field(default=True)

class MixtureDensityHead(nn.Module):
    def __init__(self, config: DictConfig, **kwargs):
        self.hparams = config
        super().__init__()
        self._build_network()

    def _build_network(self):
        self.pi = nn.Linear(self.hparams.input_dim, self.hparams.num_gaussian)
        nn.init.normal_(self.pi.weight)
        self.sigma = nn.Linear(
            self.hparams.input_dim,
            self.hparams.num_gaussian,
            bias=self.hparams.sigma_bias_flag,
        )
        self.mu = nn.Linear(self.hparams.input_dim, self.hparams.num_gaussian)
        nn.init.normal_(self.mu.weight)
        if self.hparams.mu_bias_init is not None:
            for i, bias in enumerate(self.hparams.mu_bias_init):
                nn.init.constant_(self.mu.bias[i], bias)

    def forward(self, x):
        pi = self.pi(x)
        sigma = self.sigma(x)
        # Applying modified ELU activation
        sigma = nn.ELU()(sigma) + 1 + 1e-15
        mu = self.mu(x)
        return pi, sigma, mu

    def gaussian_probability(self, sigma, mu, target, log=False):
        """Returns the probability of `target` given MoG parameters `sigma` and `mu`.

        Arguments:
            sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
                size, G is the number of Gaussians, and O is the number of
                dimensions per Gaussian.
            mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
                number of Gaussians, and O is the number of dimensions per Gaussian.
            target (BxI): A batch of target. B is the batch size and I is the number of
                input dimensions.
        Returns:
            probabilities (BxG): The probability of each point in the probability
                of the distribution in the corresponding sigma/mu index.
        """
        target = target.expand_as(sigma)
        if log:
            ret = (
                -torch.log(sigma)
                – 0.5 * LOG2PI
                – 0.5 * torch.pow((target – mu) / sigma, 2)
            )
        else:
            ret = (ONEOVERSQRT2PI / sigma) * torch.exp(
                -0.5 * ((target – mu) / sigma) ** 2
            )
        return ret  # torch.prod(ret, 2)

    def log_prob(self, pi, sigma, mu, y):
        log_component_prob = self.gaussian_probability(sigma, mu, y, log=True)
        log_mix_prob = torch.log(
            nn.functional.gumbel_softmax(pi, tau=1, dim=-1) + 1e-15
        )
        return torch.logsumexp(log_component_prob + log_mix_prob, dim=-1)

    def sample(self, pi, sigma, mu):
        """Draw samples from a MoG."""
        categorical = Categorical(pi)
        pis = categorical.sample().unsqueeze(1)
        sample = Variable(sigma.data.new(sigma.size(0), 1).normal_())
        # Gathering from the n Gaussian Distribution based on sampled indices
        sample = sample * sigma.gather(1, pis) + mu.gather(1, pis)
        return sample

    def generate_samples(self, pi, sigma, mu, n_samples=None):
        if n_samples is None:
            n_samples = self.hparams.n_samples
        samples = []
        softmax_pi = nn.functional.gumbel_softmax(pi, tau=1, dim=-1)
        assert (
            softmax_pi < 0
        ).sum().item() == 0, "pi parameter should not have negative"
        for _ in range(n_samples):
            samples.append(self.sample(softmax_pi, sigma, mu))
        samples = torch.cat(samples, dim=1)
        return samples

    def generate_point_predictions(self, pi, sigma, mu, n_samples=None):
        # Sample using n_samples and take average
        samples = self.generate_samples(pi, sigma, mu, n_samples)
        if self.hparams.central_tendency == "mean":
            y_hat = torch.mean(samples, dim=-1)
        elif self.hparams.central_tendency == "median":
            y_hat = torch.median(samples, dim=-1).values
        return y_hat
