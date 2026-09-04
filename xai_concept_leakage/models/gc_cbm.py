import copy

import sklearn.metrics
import torch
import pytorch_lightning as pl
from torchvision.models import resnet50, densenet121
from torch.nn.utils import spectral_norm
import numpy as np

import xai_concept_leakage.train.utils as utils
from xai_concept_leakage.metrics.accs import compute_accuracy
from xai_concept_leakage.metrics.mutual_information import compute_MI_score_model_training, matrix_from_tril
from xai_concept_leakage.models.gc_cem import GradientReversalFunction


################################################################################
## BASELINE MODEL
################################################################################

# We only have an adversarial loss when there we meet two conditions: use_adversarial and current_epoch > adversarial delay
#we have a lambda weight scheduler which is either linear so there is a three stages:
# 1: lambda is zero until adversarial delay then we linearly increase for a warmup period and then remain at the max

class GCConceptBottleneckModel(pl.LightningModule):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        adversarial_delay,
        max_adversarial_lambda=1,
        adversarial_scheduler = None,
        adversarial_loss_type ='gradient',
        adversarial_lambda_scheduler_warmup=None,
        use_adversarial=True,
        adv_learning_rate=0.01,
        concept_loss_weight=0.01,
        task_loss_weight=1,
        extra_dims=0,
        bool=False,
        sigmoidal_prob=True,
        sigmoidal_extra_capacity=True,
        bottleneck_nonlinear=None,
        output_latent=False,
        x2c_model=None,
        c_extractor_arch=utils.wrap_pretrained_model(resnet50),
        c2y_model=None,
        c2y_layers=None,
        cbm_optimizer="adam",
        adversarial_optimizer="adam",
        momentum=0.9,
        cbm_learning_rate=0.01,
        n_critic_steps=1,
        shared_critic=False,
        weight_decay=4e-05,
        weight_loss=None,
        task_class_weights=None,
        active_intervention_values=None,
        inactive_intervention_values=None,
        intervention_policy=None,
        output_interventions=False,
        use_concept_groups=False,
        top_k_accuracy=None,
        compute_mi_on_gpu=False
    ):
        """
        Constructs a joint Concept Bottleneck Model (CBM) as defined by
        Koh et al. 2020.

        :param int n_concepts: The number of concepts given at training time.
        :param int n_tasks: The number of output classes of the CBM.
        :param int adversarial_delay: Number of epochs after which adversary is introduced.
        :param float max_adversarial_lambda: The maximum weighting we apply to the critic
            in the CBM loss function.
        :param float adversarial_scheduler: A function that takes the current epoch and returns the current adversarial
            weight (lambda_adv). Defaults to None can be one of ['lagrange', 'linear', 'sigmoid'].
        :param int adversarial_lambda_scheduler_warmup: Number of epochs over which the penalty ramps up if we have
            a linear scheduler.
        :param bool use_adversarial: If True, enables the adversarial critic and its loss. If False,
            the model trains as a standard CBM. Defaults to True.
        :param float adv_learning_rate: The learning rate for the adversarial
            critic's optimizer. Defaults to 0.01.
        :param float task_loss_weight: Weight to be used for the final loss'
            component corresponding to the output task classification loss.
            Default is 1.

        :param int extra_dims: The number of extra unsupervised dimensions to
            include in the bottleneck. Defaults to 0.
        :param Bool bool: Whether or not we threshold concepts in the bottleneck
            to be binary. Only relevant if the bottleneck uses a sigmoidal
            activation. Defaults to False.
        :param Bool sigmoidal_prob: Whether or not to use a sigmoidal activation
            for the bottleneck's activations that are aligned with training
            concepts. Defaults to True.
        :param Bool sigmoidal_extra_capacity:  Whether or not to use a sigmoidal
            activation for the bottleneck's unsupervised activations (when
            extra_dims > 0). Defaults to True.
        :param str bottleneck_nonlinear: A valid nonlinearity name to use for
            any unsupervised extra capacity in this model (when extra_dims > 0).
            It may overwrite `sigmoidal_extra_capacity` if
            sigmoidal_extra_capacity is True. If None, then no activation will
            be used. Will be soon deprecated. It must be one of [None,
            "sigmoid", "relu", "leakyrelu"] and defaults to None.

        :param Pytorch.Module x2c_model: A valid pytorch Module used to map the
            CBM's inputs to its bottleneck layer with `n_concepts + extra_dims`
            activations. If not given, then one may provide a generator
            function via the c_extractor_arch argument.
        :param Fun[(int), Pytorch.Module] c_extractor_arch: If x2c_model is None,
            then one may provide a generator function for the input to concept
            model that takes as an input the size of the bottleneck (using
            an argument called `output_dim`) and returns a valid Pytorch Module
            that maps this CBM's inputs to the bottleneck of the requested size.
        :param Pytorch.Module c2y_model:  A valid pytorch Module used to map the
            CBM's bottleneck (with size n_concepts + extra_dims`) to `n_tasks`
            output activations (i.e., the output of the CBM).
            If not given, then a simple leaky-ReLU MLP, whose hidden
            layers have sizes `c2y_layers`, will be used.
        :param List[int] c2y_layers: List of integers defining the size of the
            hidden layers to be used in the MLP to predict classes from the
            bottleneck if c2y_model was NOT provided. If not given, then we will
            use a simple linear layer to map the bottleneck to the output
            classes.


        :param str optimizer:  The name of the optimizer to use. Must be one of
            `adam` or `sgd`. Default is `adam`.
        :param float momentum: Momentum used for optimization. Default is 0.9.
        :param float cbm_learning_rate:  Learning rate used for optimization.
            Default is 0.01.
        :param float weight_decay: The weight decay factor used during
            optimization. Default is 4e-05.
        :param List[float] weight_loss: Either None or a list with n_concepts
            elements indicating the weights assigned to each predicted concept
            during the loss computation. Could be used to improve
            performance/fairness in imbalanced datasets.
        :param List[float] task_class_weights: Either None or a list with
            n_tasks elements indicating the weights assigned to each output
            class during the loss computation. Could be used to improve
            performance/fairness in imbalanced datasets.


        :param List[float] active_intervention_values: A list of n_concepts
            values to use when positively intervening in a given concept (i.e.,
            setting concept c_i to 1 would imply setting its corresponding
            predicted concept to active_intervention_values[i]). If not given,
            then we will assume that we use `1` for all concepts. This
            parameter is important when intervening in CBMs that do not have
            sigmoidal concepts, as the intervention thresholds must then be
            inferred from their empirical training distribution.
        :param List[float] inactive_intervention_values: A list of n_concepts
            values to use when negatively intervening in a given concept (i.e.,
            setting concept c_i to 0 would imply setting its corresponding
            predicted concept to inactive_intervention_values[i]). If not given,
            then we will assume that we use `0` for all concepts. This
            parameter is important when intervening in CBMs that do not have
            sigmoidal concepts, as the intervention thresholds must then be
            inferred from their empirical training distribution.
        :param Callable[(np.ndarray, np.ndarray, np.ndarray), np.ndarray] intervention_policy:
            An optional intervention policy to be used when intervening on a
            test batch sample x (first argument), with corresponding true
            concepts c (second argument), and true labels y (third argument).
            The policy must produce as an output a list of concept indices to
            intervene (in batch form) or a batch of binary masks indicating
            which concepts we will intervene on.

        :param List[int] top_k_accuracy: List of top k values to report accuracy
            for during training/testing when the number of tasks is high.
        """
        super().__init__()
        self.n_concepts = n_concepts
        self.intervention_policy = intervention_policy
        self.output_latent = output_latent
        self.output_interventions = output_interventions
        self.use_adversarial = use_adversarial

        if x2c_model is not None:
            # Then this is assumed to be a module already provided as
            # the input to concepts method
            self.x2c_model = x2c_model
        else:
            self.x2c_model = c_extractor_arch(output_dim=(n_concepts + extra_dims))

        # Now construct the label prediction model
        if c2y_model is not None:
            # Then this method has been provided to us already
            self.c2y_model = c2y_model
        else:
            # Else we construct it here directly
            units = [n_concepts + extra_dims] + (c2y_layers or []) + [n_tasks]
            layers = []
            for i in range(1, len(units)):
                layers.append(torch.nn.Linear(units[i - 1], units[i]))
                if i != len(units) - 1:
                    layers.append(torch.nn.LeakyReLU())
            self.c2y_model = torch.nn.Sequential(*layers)

        self.cbm_params = list(self.x2c_model.parameters()) + list(self.c2y_model.parameters())
        self.shared_critic = shared_critic
        if self.use_adversarial:
            self.critic = self.c2y_model if shared_critic else copy.deepcopy(self.c2y_model)

        # Intervention-specific fields/handlers:
        if active_intervention_values is not None:
            self.active_intervention_values = torch.FloatTensor(
                active_intervention_values
            )
        else:
            # Setting to 5 for prob = 1 (as that would result in its sigmoid
            # value being very close to 1) and -5 if prob=0 (as that will
            # go to zero when applied a sigmoid)
            self.active_intervention_values = torch.FloatTensor(
                [1 for _ in range(n_concepts)]
            ) * (5.0 if not sigmoidal_prob else 1.0)
        if inactive_intervention_values is not None:
            self.inactive_intervention_values = torch.FloatTensor(
                inactive_intervention_values
            )
        else:
            # Setting to 5 for prob = 1 (as that would result in its sigmoid
            # value being very close to 1) and -5 if prob=0 (as that will
            # go to zero when applied a sigmoid)
            self.inactive_intervention_values = torch.FloatTensor(
                [1 for _ in range(n_concepts)]
            ) * (-5.0 if not sigmoidal_prob else 0.0)

        # For legacy purposes, we wrap the model around a torch.nn.Sequential
        # module
        self.sig = torch.nn.Sigmoid()
        if sigmoidal_extra_capacity:
            # Keeping this for backwards compatability
            bottleneck_nonlinear = "sigmoid"
        if bottleneck_nonlinear == "sigmoid":
            self.bottleneck_nonlin = torch.nn.Sigmoid()
        elif bottleneck_nonlinear == "leakyrelu":
            self.bottleneck_nonlin = torch.nn.LeakyReLU()
        elif bottleneck_nonlinear == "relu":
            self.bottleneck_nonlin = torch.nn.ReLU()
        elif (bottleneck_nonlinear is None) or (bottleneck_nonlinear == "identity"):
            self.bottleneck_nonlin = lambda x: x
        else:
            raise ValueError(f"Unsupported nonlinearity '{bottleneck_nonlinear}'")

        self.loss_concept = torch.nn.BCELoss(weight=weight_loss)
        self.loss_task = (
            torch.nn.CrossEntropyLoss(weight=task_class_weights)
            if n_tasks > 1
            else torch.nn.BCEWithLogitsLoss(pos_weight=task_class_weights)
        )
        self.loss_adversarial = self.loss_task
        self.adversarial_scheduler = adversarial_scheduler

        if self.use_adversarial:
            self.adversarial_delay = adversarial_delay
            self.max_adversarial_loss_weight = max_adversarial_lambda
            self.n_critic_steps = n_critic_steps
            self.adversarial_loss_type = adversarial_loss_type

            if self.adversarial_scheduler == "linear":
                #Specific to linear scheduler period over which we increase the weighting
                self.adversarial_warmup_epochs = adversarial_lambda_scheduler_warmup
            if self.adversarial_scheduler == "lagrange":
                self.lambda_scheduler = LagrangianLambdaScheduler(lr_lambda=0.1,
                                                                  init_lambda=1,
                                                                  max_lambda=self.max_adversarial_loss_weight)
        self._test_step_outputs = None
        self.bool = bool
        self.concept_loss_weight = concept_loss_weight
        self.task_loss_weight = task_loss_weight
        self.momentum = momentum
        self.cbm_learning_rate = cbm_learning_rate
        self.adv_learning_rate = adv_learning_rate
        self.weight_decay = weight_decay
        self.cbm_optimizer_name = cbm_optimizer
        self.adversarial_optimizer_name = adversarial_optimizer
        self.extra_dims = extra_dims
        self.top_k_accuracy = top_k_accuracy
        self.n_tasks = n_tasks
        self.sigmoidal_prob = sigmoidal_prob
        self.sigmoidal_extra_capacity = sigmoidal_extra_capacity
        self.use_concept_groups = use_concept_groups
        self.compute_mi_on_gpu = compute_mi_on_gpu

    def _unpack_batch(self, batch):
        x = batch[0]
        if isinstance(batch[1], list):
            y, c = batch[1]
        else:
            y, c = batch[1], batch[2]
        if len(batch) > 3:
            competencies = batch[3]
        else:
            competencies = None
        if len(batch) > 4:
            prev_interventions = batch[4]
        else:
            prev_interventions = None
        return x, y, (c, competencies, prev_interventions)

    def _standardize_indices(self, intervention_idxs, batch_size):
        if isinstance(intervention_idxs, list):
            intervention_idxs = np.array(intervention_idxs)
        if isinstance(intervention_idxs, np.ndarray):
            intervention_idxs = torch.IntTensor(intervention_idxs)

        if intervention_idxs is None or (
            isinstance(intervention_idxs, torch.Tensor)
            and ((len(intervention_idxs) == 0) or intervention_idxs.shape[-1] == 0)
        ):
            return None
        if not isinstance(intervention_idxs, torch.Tensor):
            raise ValueError(f"Unsupported intervention indices {intervention_idxs}")
        if len(intervention_idxs.shape) == 1:
            # Then we will assume that we will do use the same
            # intervention indices for the entire batch!
            intervention_idxs = torch.tile(
                torch.unsqueeze(intervention_idxs, 0),
                (batch_size, 1),
            )
        elif len(intervention_idxs.shape) == 2:
            assert intervention_idxs.shape[0] == batch_size, (
                f"Expected intervention indices to have batch size {batch_size} "
                f"but got intervention indices with "
                f"shape {intervention_idxs.shape}."
            )
        else:
            raise ValueError(
                f"Intervention indices should have 1 or 2 dimensions. Instead "
                f"we got indices with shape {intervention_idxs.shape}."
            )
        if intervention_idxs.shape[-1] == self.n_concepts:
            # We still need to check the corner case here where all indices are
            # given...
            elems = torch.unique(intervention_idxs)
            if len(elems) == 1:
                is_binary = (0 in elems) or (1 in elems)
            elif len(elems) == 2:
                is_binary = (0 in elems) and (1 in elems)
            else:
                is_binary = False
        else:
            is_binary = False
        if not is_binary:
            # Then this is an array of indices rather than a binary array!
            intervention_idxs = intervention_idxs.to(dtype=torch.long)
            result = torch.zeros(
                (batch_size, self.n_concepts),
                dtype=torch.bool,
                device=intervention_idxs.device,
            )
            result[:, intervention_idxs] = 1
            intervention_idxs = result
        assert intervention_idxs.shape[-1] == self.n_concepts, (
            f"Unsupported intervention indices with "
            f"shape {intervention_idxs.shape}."
        )
        if isinstance(intervention_idxs, np.ndarray):
            # Time to make it into a torch Tensor!
            intervention_idxs = torch.BoolTensor(intervention_idxs)
        intervention_idxs = intervention_idxs.to(dtype=torch.bool)
        return intervention_idxs


    def get_adversarial_lambda_linear(self
                                  ) -> float:
        if not self.use_adversarial or self.current_epoch < self.adversarial_delay:
            return 0.0

        current_adv_epoch = self.current_epoch - self.adversarial_delay
        warmup_fraction = min(1, current_adv_epoch/self.adversarial_warmup_epochs)

        current_adversarial_loss_weight = warmup_fraction * self.max_adversarial_loss_weight
        return current_adversarial_loss_weight

    def get_adversarial_lambda_sigmoid(self
                                       ) -> float:
        if not self.use_adversarial or self.current_epoch < self.adversarial_delay:
            return 0.0

        current_adv_epoch = self.current_epoch - self.adversarial_delay
        total_ramp_up = self.trainer.max_epochs
        ratio = current_adv_epoch/total_ramp_up
        ramp_up_speed = 10
        sigmoid_input = (ratio * 2 - 1) * (ramp_up_speed / 2)

        # Calculate the sigmoid value (which ranges from 0 to 1)
        current_adversarial_loss_weight = self.max_adversarial_loss_weight / (1 + np.exp(-sigmoid_input))
        return current_adversarial_loss_weight


    def _extra_losses(
        self,
        x,
        y,
        c,
        y_pred,
        c_sem,
        c_pred,
        competencies=None,
        prev_interventions=None,
        y_adv_pred=None
    ):
        if self.use_adversarial and self.current_epoch >= self.adversarial_delay:
            adversarial_loss = (self.loss_adversarial(
                y_adv_pred if y_adv_pred.shape[-1] > 1 else y_adv_pred.reshape(-1),
                y,))

            adversarial_loss_scalar = adversarial_loss.detach()
        else:
            adversarial_loss = torch.tensor(0.0, device=self.device)
            adversarial_loss_scalar = 0
        return adversarial_loss, adversarial_loss_scalar


    def _prior_int_distribution(
        self,
        c,
        prob,
        pos_embeddings,
        neg_embeddings,
        competencies=None,
        prev_interventions=None,
        train=False,
        horizon=1,
    ):
        return None

    def _concept_intervention(
        self,
        c_pred,
        intervention_idxs=None,
        c_true=None,
    ):
        if (c_true is None) or (intervention_idxs is None):
            return c_pred
        c_pred_copy = c_pred.clone()
        intervention_idxs = self._standardize_indices(
            intervention_idxs=intervention_idxs,
            batch_size=c_pred.shape[0],
        )
        intervention_idxs = intervention_idxs.to(c_pred.device)
        # Check whether the mask needs to be extended because of
        # extra dimensions
        if self.extra_dims:
            set_intervention_idxs = torch.nn.functional.pad(
                intervention_idxs,
                pad=(0, self.extra_dims),  # Just pads the last dimension
            )
        else:
            set_intervention_idxs = intervention_idxs
        if self.sigmoidal_prob:
            c_pred_copy[set_intervention_idxs] = c_true[intervention_idxs]
        else:
            active_intervention_values = self.active_intervention_values.to(
                c_pred.device
            )
            batched_active_intervention_values = torch.tile(
                torch.unsqueeze(active_intervention_values, 0),
                (c_pred.shape[0], 1),
            ).to(c_true.device)

            inactive_intervention_values = self.inactive_intervention_values.to(
                c_pred.device
            )
            batched_inactive_intervention_values = torch.tile(
                torch.unsqueeze(inactive_intervention_values, 0),
                (c_pred.shape[0], 1),
            ).to(c_true.device)

            c_pred_copy[set_intervention_idxs] = (
                c_true[intervention_idxs]
                * batched_active_intervention_values[intervention_idxs]
            ) + (
                (c_true[intervention_idxs] - 1)
                * -batched_inactive_intervention_values[intervention_idxs]
            )
        return c_pred_copy

    def _forward(
        self,
        x,
        intervention_idxs=None,
        competencies=None,
        prev_interventions=None,
        c=None,
        y=None,
        train=False,
        latent=None,
        output_latent=None,
        output_embeddings=False,
        output_interventions=None,
    ):
        output_interventions = (
            output_interventions
            if output_interventions is not None
            else self.output_interventions
        )
        output_latent = (
            output_latent if output_latent is not None else self.output_latent
        )
        if latent is None:
            latent = self.x2c_model(x)
        if self.sigmoidal_prob or self.bool:
            if self.extra_dims:
                # Then we only sigmoid on the probability bits but
                # let the other entries up for grabs
                c_pred_probs = self.sig(latent[:, : -self.extra_dims])
                c_others = self.bottleneck_nonlin(latent[:, -self.extra_dims :])
                c_pred = torch.cat([c_pred_probs, c_others], axis=-1)
                c_sem = c_pred_probs
            else:
                c_pred = self.sig(latent)
                c_sem = c_pred
        else:
            # Otherwise, the concept vector itself is not sigmoided
            # but the semantics
            c_pred = latent
            if self.extra_dims:
                c_sem = self.sig(latent[:, : -self.extra_dims])
            else:
                c_sem = self.sig(latent)
        if output_embeddings or (
            (intervention_idxs is None)
            and (c is not None)
            and (self.intervention_policy is not None)
        ):
            pos_embeddings = torch.ones(c_sem.shape).to(x.device)
            neg_embeddings = torch.zeros(c_sem.shape).to(x.device)
            if not (self.sigmoidal_prob or self.bool):
                if (self.active_intervention_values is not None) and (
                    self.inactive_intervention_values is not None
                ):
                    active_intervention_values = self.active_intervention_values.to(
                        c_pred.device
                    )
                    pos_embeddings = torch.tile(
                        active_intervention_values,
                        (c.shape[0], 1),
                    ).to(active_intervention_values.device)
                    inactive_intervention_values = self.inactive_intervention_values.to(
                        c_pred.device
                    )
                    neg_embeddings = torch.tile(
                        inactive_intervention_values,
                        (c.shape[0], 1),
                    ).to(inactive_intervention_values.device)
                else:
                    out_embs = c_pred.detach().cpu().numpy()
                    for concept_idx in range(self.n_concepts):
                        pos_embeddings[:, concept_idx] = np.percentile(
                            out_embs[:, concept_idx],
                            95,
                        )
                        neg_embeddings[:, concept_idx] = np.percentile(
                            out_embs[:, concept_idx],
                            5,
                        )
            pos_embeddings = torch.unsqueeze(pos_embeddings, dim=-1)
            neg_embeddings = torch.unsqueeze(neg_embeddings, dim=-1)
        # Now include any interventions that we may want to include
        if (
            (intervention_idxs is None)
            and (c is not None)
            and (self.intervention_policy is not None)
        ):
            prior_distribution = self._prior_int_distribution(
                c=c,
                prob=c_sem,
                pos_embeddings=pos_embeddings,
                neg_embeddings=neg_embeddings,
                competencies=competencies,
                prev_interventions=prev_interventions,
                train=train,
                horizon=1,
            )
            intervention_idxs, c_int = self.intervention_policy(
                x=x,
                c=c,
                pred_c=c_sem,
                y=y,
                competencies=competencies,
                prev_interventions=prev_interventions,
                prior_distribution=prior_distribution,
            )
        else:
            c_int = c
        c_pred = self._concept_intervention(
            c_pred=c_pred,
            intervention_idxs=intervention_idxs,
            c_true=c_int,
        )
        if self.bool:
            y = self.c2y_model((c_pred > 0.5).float())

        else:
            y = self.c2y_model(c_pred)


        tail_results = []
        if output_interventions:
            if intervention_idxs is None:
                intervention_idxs = None
            if isinstance(intervention_idxs, np.ndarray):
                intervention_idxs = torch.FloatTensor(intervention_idxs).to(x.device)
            tail_results.append(intervention_idxs)
        if output_latent:
            tail_results.append(latent)
        if output_embeddings:
            tail_results.append(pos_embeddings)
            tail_results.append(neg_embeddings)
        return tuple([c_sem, c_pred, y] + tail_results)

    def forward(
        self,
        x,
        c=None,
        y=None,
        latent=None,
        intervention_idxs=None,
        competencies=None,
        prev_interventions=None,
    ):
        return self._forward(
            x,
            train=False,
            c=c,
            y=y,
            competencies=competencies,
            prev_interventions=prev_interventions,
            intervention_idxs=intervention_idxs,
            latent=latent,
        )

    def setup(self, stage: str):
        if stage == 'test':
            print("--- Initializing list for test step outputs ---")
            self._test_step_outputs = []

    def predict_step(
        self,
        batch,
        batch_idx,
        intervention_idxs=None,
        dataloader_idx=0,
    ):
        x, y, (c, competencies, prev_interventions) = self._unpack_batch(batch)
        return self._forward(
            x,
            intervention_idxs=intervention_idxs,
            c=c,
            y=y,
            train=False,
            competencies=competencies,
            prev_interventions=prev_interventions,
        )

    def _monitor_grad(self,
                     batch_idx,
                     task_loss,
                     c_logits,
                     adversarial_term,
                     concept_loss,
                      train):
        if train and self.use_adversarial:
            accumulate_grad = getattr(self.trainer, "accumulate_grad_batches", 1)
            if isinstance(accumulate_grad, int):
                if accumulate_grad ==1:
                    log_frequency = 50 * accumulate_grad
                else:
                    log_frequency = 2 * accumulate_grad
            else:
                log_frequency = 50
            should_log = (batch_idx % log_frequency == 0)

            if should_log:

                grads_task=None
                if isinstance(task_loss, torch.Tensor) and task_loss.requires_grad:
                    grads_task = torch.autograd.grad(
                        task_loss, c_logits, retain_graph=True, allow_unused=True
                    )[0]

                # 2. Adversarial Gradient
                grads_adv = None
                if isinstance(adversarial_term, torch.Tensor) and adversarial_term.requires_grad:
                    grads_adv = torch.autograd.grad(
                        adversarial_term, c_logits, retain_graph=True, allow_unused=True
                    )[0]

                # 3. Concept Gradient
                grads_conc = None
                weighted_concept_loss = self.concept_loss_weight * concept_loss
                if isinstance(weighted_concept_loss, torch.Tensor) and weighted_concept_loss.requires_grad:
                    grads_conc = torch.autograd.grad(
                        weighted_concept_loss, c_logits, retain_graph=True, allow_unused=True
                    )[0]

                # --- Log Norms (How strong is the push?) ---
                if grads_task is not None:
                    self.log("grads/norm_task", grads_task.norm())
                if grads_adv is not None:
                    self.log("grads/norm_adv", grads_adv.norm())
                if grads_conc is not None:
                    self.log("grads/norm_conc", grads_conc.norm())



                # --- Log Alignment (Are they cancelling?) ---
                if grads_task is not None and grads_adv is not None:
                    g_task_flat = grads_task.view(-1)
                    g_adv_flat = grads_adv.view(-1)
                    cosine = torch.nn.functional.cosine_similarity(g_task_flat, g_adv_flat, dim=0)
                    self.log("grads/cosine_task_vs_adv", cosine)
                    residual = (g_task_flat + g_adv_flat).norm()
                    self.log("grads/norm_residual_task_adv", residual)

    def get_lambda_scheduler(self,
                         task_loss_scalar,
                        adversarial_loss_scalar):
        if self.adversarial_scheduler == 'linear':
            adversarial_loss_weight = self.get_adversarial_lambda_linear()
        elif self.adversarial_scheduler in ['lagrange', 'proportional']:
            adversarial_loss_weight = self.lambda_scheduler.update(task_loss_scalar, adversarial_loss_scalar)
        elif self.adversarial_scheduler == 'sigmoid':
            adversarial_loss_weight = self.get_adversarial_lambda_sigmoid()
        else:
            adversarial_loss_weight = self.max_adversarial_loss_weight
        self.log('current_adv_lambda', adversarial_loss_weight, on_step=False, on_epoch=True)
        return adversarial_loss_weight


    def _run_cbm_step(
        self,
        batch,
        batch_idx,
        train=False,
        intervention_idxs=None,
    ):
        x, y, (c, competencies, prev_interventions) = self._unpack_batch(batch)
        outputs = self._forward(
            x,
            intervention_idxs=intervention_idxs,
            c=c,
            y=y,
            train=train,
            competencies=competencies,
            prev_interventions=prev_interventions,
        )
        c_sem, c_logits, y_logits = outputs[0], outputs[1], outputs[2]

        if self.task_loss_weight != 0:
            task_loss = self.loss_task(
                y_logits if y_logits.shape[-1] > 1 else y_logits.reshape(-1),
                y,
            )
            task_loss_scalar = task_loss.detach()
        else:
            task_loss = 0
            task_loss_scalar = 0

        adversarial_loss_scalar = 0
        adversarial_term = 0

        if self.use_adversarial and self.adversarial_loss_type == 'grl':
            if self.current_epoch >= self.adversarial_delay:
                adversarial_loss_weight = self.get_lambda_scheduler(task_loss_scalar, 0.0)
                grl_c_logits = GradientReversalFunction.apply(
                    c_logits,
                    torch.tensor(adversarial_loss_weight, dtype=c_logits.dtype, device=c_logits.device),
                )
                y_adv_pred = self.critic(grl_c_logits)
                adversarial_term = self.loss_adversarial(
                    y_adv_pred if y_adv_pred.shape[-1] > 1 else y_adv_pred.reshape(-1),
                    y,
                )
                adversarial_loss_scalar = adversarial_term.detach().item()
        elif self.use_adversarial:
            y_adv_pred = None
            if self.current_epoch >= self.adversarial_delay:
                if self.adversarial_loss_type == 'gradient':
                    if self.bool:
                        y_adv_pred = self.critic((c_logits > 0.5).float())
                    else:
                        y_adv_pred = self.critic(c_logits)
                else:
                    #this is the accuracy gap method
                    y_adv_pred = self.critic(c)

            adversarial_loss, adversarial_loss_scalar = self._extra_losses(
                        x=x,
                        y=y,
                        c=c,
                        c_sem=c_sem,
                        c_pred=c_logits,
                        y_pred=y_logits,
                        y_adv_pred=y_adv_pred,
                        competencies=competencies,
                        prev_interventions=prev_interventions,
                    )

            adversarial_loss_weight = self.get_lambda_scheduler(task_loss_scalar, adversarial_loss_scalar)
            if self.adversarial_loss_type == 'gradient':
                adversarial_term = - (adversarial_loss_weight * adversarial_loss)
            else:
                leakage_gap = adversarial_loss - task_loss
                hinge_penalty = torch.clamp(leakage_gap, min=0)
                adversarial_term = adversarial_loss_weight * hinge_penalty

        if self.concept_loss_weight != 0:
            concept_loss = self.loss_concept(c_sem, c)
            concept_loss_scalar = concept_loss.detach().item()

            # self._monitor_grad(batch_idx,
            #          task_loss,
            #          c_logits,
            #          adversarial_term,
            #          concept_loss,
            #        train=train)
            loss = (
                self.concept_loss_weight * concept_loss
                + task_loss + adversarial_term)

        else:
            loss = task_loss + adversarial_term
            concept_loss_scalar = 0.0
        # compute accuracy
        (c_accuracy, c_auc, c_f1), (y_accuracy, y_auc, y_f1) = compute_accuracy(
            c_sem,
            y_logits,
            c,
            y,
        )
        # our adversarial loss here is scaled by the weighting already
        result = {
            "c_accuracy": c_accuracy,
            "c_auc": c_auc,
            "c_f1": c_f1,
            "y_accuracy": y_accuracy,
            "y_auc": y_auc,
            "y_f1": y_f1,
            "concept_loss": concept_loss_scalar,
            "task_loss": task_loss_scalar,
            "loss": loss.detach(),
            "adversarial_loss": adversarial_loss_scalar,
            "avg_c_y_acc": (c_accuracy + y_accuracy) / 2,
        }
        if self.top_k_accuracy is not None:
            y_true = y.reshape(-1).cpu().detach()
            y_pred = y_logits.cpu().detach()
            labels = list(range(self.n_tasks))
            if isinstance(self.top_k_accuracy, int):
                top_k_accuracy = list(range(1, self.top_k_accuracy))
            else:
                top_k_accuracy = self.top_k_accuracy

            for top_k_val in top_k_accuracy:
                if top_k_val:
                    y_top_k_accuracy = sklearn.metrics.top_k_accuracy_score(
                        y_true,
                        y_pred,
                        k=top_k_val,
                        labels=labels,
                    )
                    result[f"y_top_{top_k_val}_accuracy"] = y_top_k_accuracy
        return loss, result


    def _run_critic_step(self,
        batch,
        batch_idx,
        train=False,
        intervention_idxs=None):

        x, y, (c, competencies, prev_interventions) = self._unpack_batch(batch)
        # Freeze the CBM by using torch.no_grad()
        with torch.no_grad():
            latent = self.x2c_model(x)
            if self.sigmoidal_prob or self.bool:
                if self.extra_dims:
                    c_pred_probs = self.sig(latent[:, : -self.extra_dims])
                    c_others = self.bottleneck_nonlin(latent[:, -self.extra_dims:])
                    c_pred = torch.cat([c_pred_probs, c_others], axis=-1)
                else:
                    c_pred = self.sig(latent)
            else:
                c_pred = latent

        # Now, with gradients enabled for the critic, get its prediction
        if self.adversarial_loss_type == 'gradient':
            if self.bool:
                y_adv_logits = self.critic((c_pred > 0.5).float())
            else:
                y_adv_logits = self.critic(c_pred)
        else:
            y_adv_logits = self.critic(c)

        # Calculate the critic's loss
        critic_loss = self.loss_adversarial(
            y_adv_logits if y_adv_logits.shape[-1] > 1 else y_adv_logits.reshape(-1),
            y,
        )

        # For good observability, let's compute the critic's accuracy as well
        _, (y_adv_accuracy, y_adv_auc, _) = compute_accuracy(
            c_pred=c_pred,
            y_pred=y_adv_logits,
            c_true=c,
            y_true=y,
        )

        result = {
            "critic_loss": critic_loss.detach(),
            "critic_acc": y_adv_accuracy,
            "critic_auc": y_adv_auc,
        }
        return critic_loss, result

    def training_step(self, batch, batch_no, optimizer_idx=0):
        if self.use_adversarial and (
            self.adversarial_loss_type == 'grl' or self.shared_critic
        ):
            loss, result = self._run_cbm_step(batch, batch_no, train=True)
            for name, val in result.items():
                self.log(name, val, prog_bar=True)
            return {"loss": loss}
        if self.use_adversarial:
            if optimizer_idx == 1:
                if self.current_epoch < self.adversarial_delay:
                    return None  # Tell PyTorch Lightning to skip this optimizer step
                print("Critic Optimisation Step ...")
                critic_loss, result = self._run_critic_step(batch, batch_no)
                for name, val in result.items():
                    self.log(name, val, prog_bar=True)
                return {"loss": critic_loss}
            #do cbm optimisation step
            if optimizer_idx == 0:
                if (batch_no + 1) % self.n_critic_steps != 0:
                    return None

                print('CBM Optimisation Step ...')
                loss, result = self._run_cbm_step(batch, batch_no, train=True)
                for name, val in result.items():
                    if self.n_tasks <= 2:
                        prog_bar = (
                            ("auc" in name)
                            or ("mask_accuracy" in name)
                            or ("current_steps" in name)
                            or ("num_rollouts" in name)
                        )
                    else:
                        prog_bar = (
                            ("c_auc" in name)
                            or ("y_accuracy" in name)
                            or ("mask_accuracy" in name)
                            or ("current_steps" in name)
                            or ("num_rollouts" in name)
                        )
                    self.log(name, val, prog_bar=prog_bar)
                return {
                    "loss": loss,
                    "log": {
                        "c_accuracy": result["c_accuracy"],
                        "c_auc": result["c_auc"],
                        "c_f1": result["c_f1"],
                        "y_accuracy": result["y_accuracy"],
                        "y_auc": result["y_auc"],
                        "y_f1": result["y_f1"],
                        "concept_loss": result["concept_loss"],
                        "task_loss": result["task_loss"],
                        "adversarial_loss": result["adversarial_loss"],
                        "loss": result["loss"],
                        "avg_c_y_acc": result["avg_c_y_acc"],
                    },
                }
        else:
            loss, result = self._run_cbm_step(batch, batch_no, train=True)
            for name, val in result.items():
                if self.n_tasks <= 2:
                    prog_bar = (
                            ("auc" in name)
                            or ("mask_accuracy" in name)
                            or ("current_steps" in name)
                            or ("num_rollouts" in name)
                    )
                else:
                    prog_bar = (
                            ("c_auc" in name)
                            or ("y_accuracy" in name)
                            or ("mask_accuracy" in name)
                            or ("current_steps" in name)
                            or ("num_rollouts" in name)
                    )
                self.log(name, val, prog_bar=prog_bar)
            return {
                "loss": loss,
                "log": {
                    "c_accuracy": result["c_accuracy"],
                    "c_auc": result["c_auc"],
                    "c_f1": result["c_f1"],
                    "y_accuracy": result["y_accuracy"],
                    "y_auc": result["y_auc"],
                    "y_f1": result["y_f1"],
                    "concept_loss": result["concept_loss"],
                    "task_loss": result["task_loss"],
                    "loss": result["loss"],
                    "avg_c_y_acc": result["avg_c_y_acc"],
                },
            }

    def validation_step(self, batch, batch_no):
        _, result = self._run_cbm_step(batch, batch_no, train=False)
        for name, val in result.items():
            if self.n_tasks <= 2:
                prog_bar = "auc" in name
            else:
                prog_bar = ("c_auc" in name) or ("y_accuracy" in name)
            self.log("val_" + name, val, prog_bar=prog_bar)
        result = {"val_" + key: val for key, val in result.items()}

        x, y, (c, competencies, prev_interventions) = self._unpack_batch(batch)
        outputs = self._forward(x)
        c_sem = outputs[0]

        result["c_learnt"] = c_sem
        result["c_true"] = c
        result["y_true"] = y
        return result

    def test_step(self, batch, batch_no):
        loss, result = self._run_cbm_step(batch, batch_no, train=False)
        for name, val in result.items():
            self.log("test_" + name, val, prog_bar=True)

        x, y, (c, competencies, prev_interventions) = self._unpack_batch(batch)
        outputs = self._forward(x)
        c_sem = outputs[0]

        if self.compute_mi_on_gpu:
            device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
        else:
            device = torch.device("cpu")
        result["c_learnt"] = c_sem.detach().to(device)
        result["c_true"] = c.detach().to(device)
        result["y_true"] = y.detach().to(device)

        self._test_step_outputs.append(result)
        return result

    def on_test_epoch_end(self):
        all_c_learnt = torch.cat([out['c_learnt'] for out in self._test_step_outputs])
        all_c_truth = torch.cat([out['c_true'] for out in self._test_step_outputs])
        all_y_true = torch.cat([out['y_true'] for out in self._test_step_outputs])
        n_concepts = all_c_truth.shape[1]
        if not self.compute_mi_on_gpu:
            all_c_learnt = all_c_learnt.numpy()
            all_c_truth =all_c_truth.numpy()
            all_y_true = all_y_true.numpy()

        norm_icl = compute_MI_score_model_training(c_pred=all_c_learnt,
                                        c_true=all_c_truth,
                                        y_true=all_y_true,
                                        score_type="interconcept",
                                        wrt_true=True,
                                        n_neighbors=3,
                                        normalise=True,
                                        n_concepts=n_concepts,
                                        compute_mi_on_gpu=self.compute_mi_on_gpu
                                        )

        mean_norm_icl_i = matrix_from_tril(norm_icl).sum(axis=1) / (n_concepts - 1)
        mean_norm_icl =  mean_norm_icl_i.sum()/len(mean_norm_icl_i)

        norm_ctl_vec = compute_MI_score_model_training(c_pred=all_c_learnt,
                                        c_true=all_c_truth,
                                        y_true=all_y_true,
                                        score_type="concepts_task",
                                        wrt_true=True,
                                        n_neighbors=3,
                                        normalise=True,
                                        n_concepts=n_concepts,
                                        compute_mi_on_gpu=self.compute_mi_on_gpu
                                        )

        mean_norm_ctl = norm_ctl_vec.sum()/len(norm_ctl_vec)
        self.log('test_normalised_ctl_average', mean_norm_ctl)
        self.log('test_normalised_icl_average', mean_norm_icl)

    def configure_optimizers(self):
        if self.use_adversarial and (
            self.adversarial_loss_type == 'grl' or self.shared_critic
        ):
            if self.cbm_optimizer_name.lower() == "adam":
                optimizer = torch.optim.Adam(self.parameters(), lr=self.cbm_learning_rate, weight_decay=self.weight_decay)
            else:
                optimizer = torch.optim.SGD(self.parameters(), lr=self.cbm_learning_rate, momentum=0.9, weight_decay=self.weight_decay)
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "loss"}
        if self.use_adversarial:
            if self.cbm_optimizer_name.lower() == "adam":
                cbm_optimizer = torch.optim.Adam(
                    self.cbm_params,
                    lr=self.cbm_learning_rate,
                    weight_decay=self.weight_decay,
                )
            elif self.cbm_optimizer_name.lower() == "adamw":
                cbm_optimizer = torch.optim.AdamW(
                    self.cbm_params,
                    lr=self.cbm_learning_rate,
                    weight_decay=self.weight_decay,
                    amsgrad=True,
                    eps=1e-7
                )

            else:
                cbm_optimizer = torch.optim.SGD(
                    filter(lambda p: p.requires_grad, self.cbm_params),
                    lr=self.cbm_learning_rate,
                    momentum=self.momentum,
                    weight_decay=self.weight_decay,
                )


            if self.adversarial_optimizer_name.lower() == "adam":
                adv_optimizer = torch.optim.Adam(
                    self.critic.parameters(),
                    lr=self.adv_learning_rate,
                    weight_decay=self.weight_decay,
                )
            elif self.adversarial_optimizer_name.lower() == "adamw":
                adv_optimizer = torch.optim.AdamW(
                    self.cbm_params,
                    lr=self.cbm_learning_rate,
                    weight_decay=self.weight_decay,
                    amsgrad=True,
                    eps=1e-7
                )
            else:
                adv_optimizer = torch.optim.SGD(
                    filter(lambda p: p.requires_grad, self.critic.parameters()),
                    lr=self.adv_learning_rate,
                    momentum=self.momentum,
                    weight_decay=self.weight_decay,
                )

            return [cbm_optimizer, adv_optimizer]


        else:
            if self.cbm_optimizer_name.lower() == "adam":
                cbm_optimizer = torch.optim.Adam(
                    self.parameters(),
                    lr=self.cbm_learning_rate,
                    weight_decay=self.weight_decay,
                )
            elif self.cbm_optimizer_name.lower() == "adamw":
                cbm_optimizer = torch.optim.AdamW(
                    self.cbm_params,
                    lr=self.cbm_learning_rate,
                    weight_decay=self.weight_decay,
                    amsgrad=True,
                    eps=1e-7
                )
            else:
                cbm_optimizer = torch.optim.SGD(
                    filter(lambda p: p.requires_grad, self.parameters()),
                    lr=self.cbm_learning_rate,
                    momentum=self.momentum,
                    weight_decay=self.weight_decay,
                )


            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                cbm_optimizer,
            )

            return {
                "optimizer": cbm_optimizer,
                "lr_scheduler": lr_scheduler,
                "monitor": "loss",
            }


class ProportionalLambdaScheduler:
    """A smoothed dynamic scheduler for the weighing of our adversarial loss by tracking the EMA of the (classifier_loss - adversarial_loss)/adversarial_loss ratio

    This acts like a PID controller by:
    1. Tracking the 'gap' (the error signal).
    2. Using the 'loss_task' to auto-scale the gain (auto-warmup).
    3. Using Exponential Moving Average (EMA) to smooth the signal.
    """
    def __init__(self,
                 beta,
                 scale_factor,
                 max_lambda):
        self.beta = beta
        self.scale_factor = scale_factor
        self.max_lambda = max_lambda
        self.ema_loss_task = None
        self.ema_loss_adv = None
        self.current_lambda = 0

    def update(self, current_task_loss,
            current_adversarial_loss):

        if self.ema_loss_task is None:
            self.ema_loss_task = current_task_loss
            self.ema_loss_adv = current_adversarial_loss

        else:
            self.ema_loss_task = (self.beta * self.ema_loss_task) + (1- self.beta) * current_task_loss
            self.ema_loss_adv = (self.beta * self.ema_loss_adv) + (1- self.beta) * current_adversarial_loss

        ema_gap = self.ema_loss_task - self.ema_loss_adv
        error_signal = max(0, ema_gap) / (self.ema_loss_task + 1e-8)

        scaled_lambda = error_signal * self.scale_factor
        self.current_lambda = np.clip(scaled_lambda, 0, self.max_lambda)
        return self.current_lambda

    def get_lambda(self):
        return self.current_lambda



class LagrangianLambdaScheduler:
    """A smoothed dynamic scheduler for the weighing of our adversarial loss using Integral Controller based principles.
    Lambda is a learnable Lagrangian multiplier.
    lambda_{t+1} = lambda_t + lr * Gap_t
    This acts like a PID controller by:
    1. Tracking the 'gap' (the error signal).
    2. Using the 'loss_task' to auto-scale the gain (auto-warmup).
    3. Using Exponential Moving Average (EMA) to smooth the signal.
    """
    def __init__(self,
                 init_lambda=1,
                 lr_lambda = 0.01,
                 target_margin = 0.0,
                 max_lambda=1.2):
        self.current_lambda = init_lambda
        self.lr = lr_lambda
        self.target_margin = target_margin
        self.log_lambda = torch.nn.Parameter(torch.tensor(np.log(max(1e-6, init_lambda))))
        self.log_lambda.requires_grad = False
        self.max_lambda = max_lambda

    def update(self, current_task_loss,
            current_adversarial_loss):

        loss_gap = (current_task_loss + self.target_margin) - current_adversarial_loss
        with torch.no_grad():
            self.current_lambda += self.lr * loss_gap
            self.current_lambda = max(0.0, min(self.max_lambda, self.current_lambda))
        return self.current_lambda

    def get_lambda(self):
        return self.current_lambda


# Backwards compatibility aliases
AdversarialConceptBottleneckModel = GCConceptBottleneckModel
CriticRegularisedConceptBottleneckModel = GCConceptBottleneckModel