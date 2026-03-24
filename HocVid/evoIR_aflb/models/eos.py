"""
EOS — Evolutionary Optimization Strategy (EvoIR Stage 1)

Converted from eos.ipynb with fix:
- MS-SSIM: kernel_size=11, betas=(0.0448, 0.2856, 0.3001) for better stability
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
from typing import List, Tuple, Optional
import numpy as np

try:
    from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
    HAS_TORCHMETRICS = True
except ImportError:
    HAS_TORCHMETRICS = False


# ===================== MS-SSIM Loss =====================

class MS_SSIMLoss(nn.Module):
    """
    Multi-Scale Structural Similarity Loss.
    FIX: kernel_size=11 and reduced betas for small patches.
    """
    def __init__(self, data_range=1.0):
        super().__init__()
        if HAS_TORCHMETRICS:
            self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(
                data_range=data_range,
                kernel_size=11,  # FIX: was 5, increased to 11
                betas=(0.0448, 0.2856, 0.3001)  # FIX: standard 3-scale betas
            )
        else:
            self.ms_ssim = None

    def forward(self, pred, target):
        if self.ms_ssim is not None:
            device = pred.device
            self.ms_ssim = self.ms_ssim.to(device)
            return 1 - self.ms_ssim(pred, target)
        else:
            return F.mse_loss(pred, target)


# ===================== Population Init =====================

def init_population(size: int = 5) -> List[List[float]]:
    """Initialize population of [λ_L1, λ_SSIM] on simplex."""
    population = []
    for _ in range(size):
        lambda1 = random.uniform(0.1, 0.9)
        lambda2 = 1.0 - lambda1
        population.append([lambda1, lambda2])
    return population


# ===================== Fitness Evaluation =====================

def evaluate_individual_from_cache(
    individual: List[float],
    restored_list: List[torch.Tensor],
    clean_list: List[torch.Tensor],
    l1_loss: nn.Module,
    ms_ssim_loss: nn.Module
) -> float:
    """Evaluate fitness on cached pairs. Returns negative loss (higher=better)."""
    total_loss = 0.0
    lambda1, lambda2 = individual
    with torch.no_grad():
        for restored, clean in zip(restored_list, clean_list):
            if restored.dim() == 3:
                restored = restored.unsqueeze(0)
                clean = clean.unsqueeze(0)
            loss = lambda1 * l1_loss(restored, clean) + lambda2 * ms_ssim_loss(restored, clean)
            total_loss += loss.item()
    return -total_loss


# ===================== Crossover & Mutation =====================

def crossover(p1: List[float], p2: List[float]) -> Tuple[List[float], List[float]]:
    """Convex combination crossover on simplex."""
    alpha = random.random()
    child1 = [alpha * p1[0] + (1 - alpha) * p2[0], alpha * p1[1] + (1 - alpha) * p2[1]]
    child2 = [alpha * p2[0] + (1 - alpha) * p1[0], alpha * p2[1] + (1 - alpha) * p1[1]]
    return child1, child2


def mutate(individual: List[float], mutation_rate: float = 0.1) -> List[float]:
    """Mutate with small perturbation, keeping on simplex."""
    if random.random() < mutation_rate:
        delta = random.uniform(-0.1, 0.1)
        individual[0] = min(max(individual[0] + delta, 0.1), 0.9)
        individual[1] = 1.0 - individual[0]
    return individual


# ===================== Evolution Loop =====================

def evolve_loss_weights_from_cache(
    restored_list: List[torch.Tensor],
    clean_list: List[torch.Tensor],
    l1_loss: nn.Module,
    ms_ssim_loss: nn.Module,
    generations: int = 3,
    pop_size: int = 5
) -> List[float]:
    """Full EOS: evolve optimal loss weights."""
    population = init_population(pop_size)
    best = None
    best_score = -float('inf')

    for gen in range(generations):
        scores = []
        for ind in population:
            score = evaluate_individual_from_cache(
                ind, restored_list, clean_list, l1_loss, ms_ssim_loss)
            scores.append((score, ind))
        scores.sort(reverse=True)
        gen_best_score, gen_best = scores[0]
        if gen_best_score > best_score:
            best_score = gen_best_score
            best = gen_best[:]

        next_population = [gen_best[:]]
        while len(next_population) < pop_size:
            p1, p2 = random.sample(scores[:3], 2)
            child1, child2 = crossover(p1[1], p2[1])
            next_population.append(mutate(child1[:]))
            next_population.append(mutate(child2[:]))
        population = next_population[:pop_size]

    return best


# ===================== EMA Teacher =====================

class EMATeacher:
    """Exponential Moving Average Teacher: θ_T ← m·θ_T + (1-m)·θ_S."""
    def __init__(self, student_model: nn.Module, momentum: float = 0.999):
        self.momentum = momentum
        self.teacher_model = copy.deepcopy(student_model)
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()

    @torch.no_grad()
    def update(self, student_model: nn.Module):
        for t_param, s_param in zip(
            self.teacher_model.parameters(), student_model.parameters()
        ):
            t_param.data.mul_(self.momentum).add_(s_param.data, alpha=1.0 - self.momentum)

    @torch.no_grad()
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        self.teacher_model.eval()
        return self.teacher_model(x)

    def state_dict(self):
        return self.teacher_model.state_dict()

    def load_state_dict(self, state_dict):
        self.teacher_model.load_state_dict(state_dict)


# ===================== EOS Manager =====================

class EOSManager:
    """Complete EOS manager with EMA teacher and periodic evolution."""
    def __init__(self, student_model, ema_momentum=0.999, evolution_interval=500,
                 pop_size=5, generations=3, initial_lambda=None):
        self.ema_teacher = EMATeacher(student_model, ema_momentum)
        self.evolution_interval = evolution_interval
        self.pop_size = pop_size
        self.generations = generations

        if initial_lambda is None:
            initial_lambda = [0.8, 0.2]
        self.lambda_l1 = initial_lambda[0]
        self.lambda_ssim = initial_lambda[1]

        self.l1_loss = nn.L1Loss()
        self.ms_ssim_loss = MS_SSIMLoss(data_range=1.0)

        self.restored_cache = []
        self.clean_cache = []
        self.step_count = 0
        self.weight_history = [(self.lambda_l1, self.lambda_ssim)]

    def compute_loss(self, restored, clean):
        return (self.lambda_l1 * self.l1_loss(restored, clean) +
                self.lambda_ssim * self.ms_ssim_loss(restored, clean))

    def step(self, student_model, restored, clean):
        self.step_count += 1
        self.ema_teacher.update(student_model)

        self.restored_cache.append(restored.detach().cpu())
        self.clean_cache.append(clean.detach().cpu())

        # FIX: Clip cache on EVERY step (not only after evolution)
        if len(self.restored_cache) > 32:
            self.restored_cache = self.restored_cache[-32:]
            self.clean_cache = self.clean_cache[-32:]

        if self.step_count % self.evolution_interval == 0:
            self._evolve()
            return True
        return False

    def _evolve(self):
        if len(self.restored_cache) < 2:
            return
        new_weights = evolve_loss_weights_from_cache(
            self.restored_cache, self.clean_cache,
            self.l1_loss, self.ms_ssim_loss,
            generations=self.generations, pop_size=self.pop_size)
        self.lambda_l1, self.lambda_ssim = new_weights
        self.weight_history.append((self.lambda_l1, self.lambda_ssim))
        self.restored_cache = []
        self.clean_cache = []

    def get_weights(self):
        return self.lambda_l1, self.lambda_ssim

    def get_history(self):
        return self.weight_history
