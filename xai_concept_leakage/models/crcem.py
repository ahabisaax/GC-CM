import copy

import numpy as np
import pytorch_lightning as pl
import torch
import sklearn.metrics
import torch.nn.functional as F

from torchvision.models import resnet50

from xai_concept_leakage.metrics.accs import compute_accuracy
from xai_concept_leakage.models.cbm import ConceptBottleneckModel
import xai_concept_leakage.train.utils as utils
from xai_concept_leakage.metrics.accs import compute_accuracy
from xai_concept_leakage.metrics.mutual_information import matrix_from_tril, compute_MI_score_model_training



################################################################################
## OUR MODEL
################################################################################


class CriticRegularisedConceptEmbeddingModel(ConceptBottleneckModel):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        n_hidden=128,
        emb_size=16,
        training_intervention_prob=0.25,
        embedding_activation="leakyrelu",
        shared_prob_gen=True,
        concept_loss_weight=1,
        task_loss_weight=1,
        c2y_model=None,
        c2y_layers=None,
        c_extractor_arch=utils.wrap_pretrained_model(resnet50),
        output_latent=False,
        cbm_optimizer="adam",
        adversarial_optimizer="adam",
        momentum=0.9,
        cem_learning_rate=0.01,
        weight_decay=4e-05,
        weight_loss=None,
        task_class_weights=None,
        tau=1,
        active_intervention_values=None,
        inactive_intervention_values=None,
        intervention_policy=None,
        output_interventions=False,
        use_concept_groups=False,
        top_k_accuracy=None,
        use_adversarial=True,
        max_adversarial_lambda=1,
        adversarial_delay=0,
        adversarial_scheduler=None,
        adversarial_loss_type='gradient',
        adversarial_lambda_scheduler_warmup=None,
        adv_learning_rate=0.01,
        n_critic_steps=1,
        compute_mi_on_gpu=False,
        concept_vector_regularisation=True,
        entropy_maximisation=True
    ):
        """
        Constructs a Concept Embedding Model (CEM) as defined by
        Espinosa Zarlenga et al. 2022.

        :param int n_concepts: The number of concepts given at training time.
        :param int n_tasks: The number of output classes of the CEM.
        :param int emb_size: The size of each concept embedding. Defaults to 16.
        :param float training_intervention_prob: RandInt probability. Defaults
            to 0.25.
        :param str embedding_activation: A valid nonlinearity name to use for the
            generated embeddings. It must be one of [None, "sigmoid", "relu",
            "leakyrelu"] and defaults to "leakyrelu".
        :param Bool shared_prob_gen: Whether or not weights are shared across
            all probability generators. Defaults to True.
        :param float concept_loss_weight: Weight to be used for the final loss'
            component corresponding to the concept classification loss. Default
            is 0.01.
        :param float task_loss_weight: Weight to be used for the final loss'
            component corresponding to the output task classification loss.
            Default is 1.

        :param Pytorch.Module c2y_model:  A valid pytorch Module used to map the
            CEM's bottleneck (with size n_concepts * emb_size) to `n_tasks`
            output activations (i.e., the output of the CEM).
            If not given, then a simple leaky-ReLU MLP, whose hidden
            layers have sizes `c2y_layers`, will be used.
        :param List[int] c2y_layers: List of integers defining the size of the
            hidden layers to be used in the MLP to predict classes from the
            bottleneck if c2y_model was NOT provided. If not given, then we will
            use a simple linear layer to map the bottleneck to the output classes.
        :param Fun[(int), Pytorch.Module] c_extractor_arch: A generator function
            for the latent code generator model that takes as an input the size
            of the latent code before the concept embedding generators act (
            using an argument called `output_dim`) and returns a valid Pytorch
            Module that maps this CEM's inputs to the latent space of the
            requested size.

        :param str cem_optimizer:  The name of the optimizer to use. Must be one of
            `adam` or `sgd`. Default is `adam`.
        :param float momentum: Momentum used for optimization. Default is 0.9.
        :param float cem_learning_rate:  Learning rate used for optimization.
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
            parameter is important when intervening in CEMs that do not have
            sigmoidal concepts, as the intervention thresholds must then be
            inferred from their empirical training distribution.
        :param List[float] inactive_intervention_values: A list of n_concepts
            values to use when negatively intervening in a given concept (i.e.,
            setting concept c_i to 0 would imply setting its corresponding
            predicted concept to inactive_intervention_values[i]). If not given,
            then we will assume that we use `0` for all concepts.
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
        pl.LightningModule.__init__(self)
        self.n_hidden = n_hidden
        self.n_concepts = n_concepts
        self.output_interventions = output_interventions
        self.intervention_policy = intervention_policy
        self.pre_concept_model = c_extractor_arch(output_dim=n_hidden)
        self.training_intervention_prob = training_intervention_prob
        self.output_latent = output_latent
        self.concept_regularisation = concept_vector_regularisation
        self.entropy_maximisation= entropy_maximisation
        if self.training_intervention_prob != 0:
            self.ones = torch.ones(n_concepts)

        if active_intervention_values is not None:
            self.active_intervention_values = torch.tensor(active_intervention_values)
        else:
            self.active_intervention_values = torch.ones(n_concepts)
        if inactive_intervention_values is not None:
            self.inactive_intervention_values = torch.tensor(
                inactive_intervention_values
            )
        else:
            self.inactive_intervention_values = torch.ones(n_concepts)
        self.task_loss_weight = task_loss_weight
        self.concept_context_generators = torch.nn.ModuleList()
        self.concept_prob_generators = torch.nn.ModuleList()
        self.shared_prob_gen = shared_prob_gen
        self.top_k_accuracy = top_k_accuracy
        for i in range(n_concepts):
            if embedding_activation is None:
                self.concept_context_generators.append(
                    torch.nn.Sequential(
                        *[
                            torch.nn.Linear(
                                self.n_hidden,
                                # list(
                                #     self.pre_concept_model.modules()
                                # )[-1].out_features,
                                # Two as each concept will have a positive and a
                                # negative embedding portion which are later mixed
                                2 * emb_size,
                            ),
                        ]
                    )
                )
            elif embedding_activation == "sigmoid":
                self.concept_context_generators.append(
                    torch.nn.Sequential(
                        *[
                            torch.nn.Linear(
                                self.n_hidden,
                                # list(
                                #     self.pre_concept_model.modules()
                                # )[-1].out_features,
                                # Two as each concept will have a positive and a
                                # negative embedding portion which are later mixed
                                2 * emb_size,
                            ),
                            torch.nn.Sigmoid(),
                        ]
                    )
                )
            elif embedding_activation == "leakyrelu":
                self.concept_context_generators.append(
                    torch.nn.Sequential(
                        *[
                            torch.nn.Linear(
                                self.n_hidden,
                                # list(
                                #     self.pre_concept_model.modules()
                                # )[-1].out_features,
                                # Two as each concept will have a positive and a
                                # negative embedding portion which are later mixed
                                2 * emb_size,
                            ),
                            torch.nn.LeakyReLU(),
                        ]
                    )
                )
            elif embedding_activation == "relu":
                self.concept_context_generators.append(
                    torch.nn.Sequential(
                        *[
                            torch.nn.Linear(
                                self.n_hidden,
                                # list(
                                #     self.pre_concept_model.modules()
                                # )[-1].out_features,
                                # Two as each concept will have a positive and a
                                # negative embedding portion which are later mixed
                                2 * emb_size,
                            ),
                            torch.nn.ReLU(),
                        ]
                    )
                )
            if self.shared_prob_gen and (len(self.concept_prob_generators) == 0):
                # Then we will use one and only one probability generator which
                # will be shared among all concepts. This will force concept
                # embedding vectors to be pushed into the same latent space
                self.concept_prob_generators.append(
                    torch.nn.Linear(
                        2 * emb_size,
                        1,
                    )
                )
            elif not self.shared_prob_gen:
                self.concept_prob_generators.append(
                    torch.nn.Linear(
                        2 * emb_size,
                        1,
                    )
                )
        if c2y_model is None:
            # Else we construct it here directly
            units = [n_concepts * emb_size] + (c2y_layers or []) + [n_tasks]
            layers = []
            for i in range(1, len(units)):
                layers.append(torch.nn.Linear(units[i - 1], units[i]))
                if i != len(units) - 1:
                    layers.append(torch.nn.LeakyReLU())
            self.c2y_model = torch.nn.Sequential(*layers)
        else:
            self.c2y_model = c2y_model



        self.critic = copy.deepcopy(self.c2y_model)
        self.sig = torch.nn.Sigmoid()

        self.loss_concept = torch.nn.BCELoss(weight=weight_loss)
        self.loss_task = (
            torch.nn.CrossEntropyLoss(weight=task_class_weights)
            if n_tasks > 1
            else torch.nn.BCEWithLogitsLoss(weight=task_class_weights)
        )
        self.concept_loss_weight = concept_loss_weight
        self.momentum = momentum
        self.cem_learning_rate = cem_learning_rate
        self.weight_decay = weight_decay
        self.cem_optimizer_name = cbm_optimizer
        self.n_tasks = n_tasks
        self.emb_size = emb_size
        self.tau = tau
        self.use_concept_groups = use_concept_groups
        self._test_step_outputs = None
        self.compute_mi_on_gpu = compute_mi_on_gpu
        self.use_adversarial = use_adversarial
        self.loss_adversarial = self.loss_task
        self.adversarial_delay = adversarial_delay
        self.adversarial_scheduler = adversarial_scheduler
        self.adversarial_loss_type = adversarial_loss_type
        self.adversarial_lambda_scheduler_warmup = adversarial_lambda_scheduler_warmup
        self.adv_learning_rate = adv_learning_rate
        self.n_critic_steps = n_critic_steps
        self.adversarial_optimizer_name = adversarial_optimizer
        self.max_adversarial_loss_weight = max_adversarial_lambda
        self.cem_params = (
                    list(self.pre_concept_model.parameters()) +
                    list(self.concept_context_generators.parameters()) +
                    list(self.concept_prob_generators.parameters()) +
                    list(self.c2y_model.parameters())
                )



    def _after_interventions(
        self,
        prob,
        pos_embeddings,
        neg_embeddings,
        intervention_idxs=None,
        c_true=None,
        train=False,
        competencies=None,
    ):
        if (
            train
            and (self.training_intervention_prob != 0)
            and ((c_true is not None) and (intervention_idxs is None))
        ):
            # Then we will probabilistically intervene in some concepts
            mask = torch.bernoulli(
                self.ones * self.training_intervention_prob,
            )
            intervention_idxs = torch.tile(
                mask,
                (c_true.shape[0], 1),
            )
        if (c_true is None) or (intervention_idxs is None):
            return prob, intervention_idxs
        intervention_idxs = intervention_idxs.type(torch.FloatTensor)
        intervention_idxs = intervention_idxs.to(prob.device)
        return (
            prob * (1 - intervention_idxs) + intervention_idxs * c_true,
            intervention_idxs,
        )

    def _forward(
        self,
        x,
        intervention_idxs=None,
        c=None,
        y=None,
        train=False,
        latent=None,
        competencies=None,
        prev_interventions=None,
        output_embeddings=False,
        output_latent=True,
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
            pre_c = self.pre_concept_model(x)
            contexts = []
            c_sem = []

            # First predict all the concept probabilities
            for i, context_gen in enumerate(self.concept_context_generators):
                if self.shared_prob_gen:
                    prob_gen = self.concept_prob_generators[0]
                else:
                    prob_gen = self.concept_prob_generators[i]
                context = context_gen(pre_c)
                prob = prob_gen(context)
                contexts.append(torch.unsqueeze(context, dim=1))
                c_sem.append(self.sig(prob))
            c_sem = torch.cat(c_sem, axis=-1)
            contexts = torch.cat(contexts, axis=1)
            latent = contexts, c_sem
        else:
            contexts, c_sem = latent


        # Now include any interventions that we may want to perform!
        if (
            (intervention_idxs is None)
            and (c is not None)
            and (self.intervention_policy is not None)
        ):
            horizon = self.intervention_policy.num_groups_intervened
            if hasattr(self.intervention_policy, "horizon"):
                horizon = self.intervention_policy.horizon
            prior_distribution = self._prior_int_distribution(
                prob=c_sem,
                pos_embeddings=contexts[:, :, : self.emb_size],
                neg_embeddings=contexts[:, :, self.emb_size :],
                competencies=competencies,
                prev_interventions=prev_interventions,
                c=c,
                train=train,
                horizon=horizon,
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
        if not train:
            intervention_idxs = self._standardize_indices(
                intervention_idxs=intervention_idxs,
                batch_size=x.shape[0],
            )

        # Then, time to do the mixing between the positive and the
        # negative embeddings
        probs, intervention_idxs = self._after_interventions(
            c_sem,
            pos_embeddings=contexts[:, :, : self.emb_size],
            neg_embeddings=contexts[:, :, self.emb_size :],
            intervention_idxs=intervention_idxs,
            c_true=c_int,
            train=train,
            competencies=competencies,
        )
        # Then time to mix!
        c_pred = contexts[:, :, : self.emb_size] * torch.unsqueeze(
            probs, dim=-1
        ) + contexts[:, :, self.emb_size :] * (1 - torch.unsqueeze(probs, dim=-1))
        c_pred = c_pred.view((-1, self.emb_size * self.n_concepts))
        y = self.c2y_model(c_pred)
        tail_results = []
        if output_interventions:
            if (intervention_idxs is not None) and isinstance(
                intervention_idxs, np.ndarray
            ):
                intervention_idxs = torch.FloatTensor(intervention_idxs).to(x.device)
            tail_results.append(intervention_idxs)
        if output_latent:
            tail_results.append(latent)
        if output_embeddings:
            tail_results.append(contexts[:, :, : self.emb_size])
            tail_results.append(contexts[:, :, self.emb_size :])
        return tuple([c_sem, c_pred, y] + tail_results)

    def setup(self, stage: str):
        if stage == 'test':
            self._test_step_outputs = []

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
            if self.entropy_maximisation:
                num_classes = y_adv_pred.size(-1)
                uniform_targets = torch.full_like(y_adv_pred, fill_value=1.0 / num_classes)
                adversarial_loss = - F.cross_entropy(y_adv_pred, uniform_targets)

            else:
                adversarial_loss = (self.loss_adversarial(
                    y_adv_pred if y_adv_pred.shape[-1] > 1 else y_adv_pred.reshape(-1),
                    y,))

            adversarial_loss_scalar = adversarial_loss.detach()
        else:
            adversarial_loss = torch.tensor(0.0, device=self.device)
            adversarial_loss_scalar = 0
        return adversarial_loss, adversarial_loss_scalar

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


    def _run_cem_step(
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
        contexts, c_sem  = outputs[3]
        probs, intervention_idxs = self._after_interventions(
            c_sem,
            pos_embeddings=contexts[:, :, : self.emb_size],
            neg_embeddings=contexts[:, :, self.emb_size:],
            intervention_idxs=intervention_idxs,
            c_true=c,
            train=train,
            competencies=competencies,
        )
        # Then time to mix!
        c_pred = contexts[:, :, : self.emb_size] * torch.unsqueeze(
            probs, dim=-1
        ) + contexts[:, :, self.emb_size:] * (1 - torch.unsqueeze(probs, dim=-1))
        c_pred = c_pred.view((-1, self.emb_size * self.n_concepts))
        y_adv_pred = None
        if self.use_adversarial and self.current_epoch >= self.adversarial_delay:
            if self.adversarial_loss_type == 'gradient':
                y_adv_pred = self.critic(c_pred)
            else:
                y_adv_pred = self.critic(c)

        adversarial_loss, adversarial_loss_scalar = self._extra_losses(
                    x=x,
                    y=y,
                    c=c,
                    c_sem=c_sem,
                    c_pred=c_logits,
                    y_pred=y_logits,
                    y_adv_pred = y_adv_pred,
                    competencies=competencies,
                    prev_interventions=prev_interventions,
                )

        if self.task_loss_weight != 0:
            task_loss = self.loss_task(
                y_logits if y_logits.shape[-1] > 1 else y_logits.reshape(-1),
                y,
            )
            task_loss_scalar = task_loss.detach()
        else:
            task_loss = 0
            task_loss_scalar = 0

        if self.use_adversarial:
            adversarial_loss_weight = self.get_lambda_scheduler(task_loss_scalar, adversarial_loss_scalar)
            if self.adversarial_loss_type == 'gradient':
                adversarial_term = - (adversarial_loss_weight * adversarial_loss)
            else:
                leakage_gap = adversarial_loss - task_loss
                hinge_penalty = torch.clamp(leakage_gap, min=0)
                adversarial_term = adversarial_loss_weight * hinge_penalty
        else:
            adversarial_term = 0

        if self.concept_loss_weight != 0:
            concept_loss = self.loss_concept(c_sem, c)
            concept_loss_scalar = concept_loss.detach().item()
            self._monitor_grad(batch_idx,
                     task_loss,
                     c_logits,
                     adversarial_term,
                     concept_loss,
                   train=train)
            loss = (
                self.concept_loss_weight * concept_loss
                + task_loss + adversarial_term)

            if self.concept_regularisation:
                l2_penalty = (contexts ** 2).mean()
                loss += 1e-2 * l2_penalty

        else:
            loss = task_loss + adversarial_term
            if self.concept_regularisation:
                l2_penalty = (contexts ** 2).mean()
                loss += 1e-2 * l2_penalty

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
        outputs = self._forward(
            x,
            intervention_idxs=intervention_idxs,
            c=c,
            y=y,
            train=train,
            competencies=competencies,
            prev_interventions=prev_interventions,
        )
        c_sem, c_pred, y_logits = outputs[0], outputs[1], outputs[2]
        contexts, c_sem = outputs[3]
        probs, intervention_idxs = self._after_interventions(
            c_sem,
            pos_embeddings=contexts[:, :, : self.emb_size],
            neg_embeddings=contexts[:, :, self.emb_size:],
            intervention_idxs=intervention_idxs,
            c_true=c,
            train=train,
            competencies=competencies,
        )
        # Then time to mix!
        c_pred = contexts[:, :, : self.emb_size] * torch.unsqueeze(
            probs, dim=-1
        ) + contexts[:, :, self.emb_size:] * (1 - torch.unsqueeze(probs, dim=-1))
        c_pred = c_pred.view((-1, self.emb_size * self.n_concepts))

        # Now, with gradients enabled for the critic, get its prediction
        if self.adversarial_loss_type == 'gradient':
            # if self.bool:
            #     y_adv_logits = self.critic((c_pred > 0.5).float())
            # else:
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
        if self.use_adversarial:
            if optimizer_idx == 1:
                if self.current_epoch < self.adversarial_delay:
                    return None  # Tell PyTorch Lightning to skip this optimizer step
                print("Critic Optimisation Step ...")
                critic_loss, result = self._run_critic_step(batch, batch_no)
                for name, val in result.items():
                    self.log(name, val, prog_bar=True)
                return {"loss": critic_loss}

            #do cem optimisation step
            if optimizer_idx == 0:
                if (batch_no + 1) % self.n_critic_steps != 0:
                    return None

                print('CEM Optimisation Step ...')
                loss, result = self._run_cem_step(batch, batch_no, train=True)
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
            loss, result = self._run_cem_step(batch, batch_no, train=True)
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
        _, result = self._run_cem_step(batch, batch_no, train=False)
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
        loss, result = self._run_cem_step(batch, batch_no, train=False)
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
        if self.use_adversarial:
            if self.cem_optimizer_name.lower() == "adam":
                cem_optimizer = torch.optim.Adam(
                    self.cem_params,
                    lr=self.cem_learning_rate,
                    weight_decay=self.weight_decay,
                )
            elif self.cem_optimizer_name.lower() == "adamw":
                cem_optimizer = torch.optim.AdamW(
                    self.cem_params,
                    lr=self.cem_learning_rate,
                    weight_decay=self.weight_decay,
                    amsgrad=True,
                    eps=1e-7
                )

            else:
                cem_optimizer = torch.optim.SGD(
                    filter(lambda p: p.requires_grad, self.cem_params),
                    lr=self.cem_learning_rate,
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
                    self.critic.parameters(),
                    lr=self.adv_learning_rate,
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

            return [cem_optimizer, adv_optimizer]


        else:
            if self.cem_optimizer_name.lower() == "adam":
                cem_optimizer = torch.optim.Adam(
                    self.parameters(),
                    lr=self.cem_learning_rate,
                    weight_decay=self.weight_decay,
                )
            elif self.cem_optimizer_name.lower() == "adamw":
                cem_optimizer = torch.optim.AdamW(
                    self.cem_params,
                    lr=self.cem_learning_rate,
                    weight_decay=self.weight_decay,
                    amsgrad=True,
                    eps=1e-7
                )
            else:
                cem_optimizer = torch.optim.SGD(
                    filter(lambda p: p.requires_grad, self.parameters()),
                    lr=self.cem_learning_rate,
                    momentum=self.momentum,
                    weight_decay=self.weight_decay,
                )


            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                cem_optimizer,
                verbose=True,
            )

            return {
                "optimizer": cem_optimizer,
                "lr_scheduler": lr_scheduler,
                "monitor": "loss",
            }
