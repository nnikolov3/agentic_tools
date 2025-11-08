import math
from datetime import datetime, timezone
from typing import Optional

class RFMCalculator:
    """
    RFM (Recency, Frequency, Monetary) calculator for memory prioritization.
    Based on Ebbinghaus forgetting curve and marketing science.
    """

    def __init__(
        self,
        recency_half_life_days: float = 30.0,
        frequency_max_accesses: int = 100,
        frequency_log_base: float = 10.0
    ):
        self.recency_half_life_seconds = recency_half_life_days * 86400
        self.frequency_max_accesses = frequency_max_accesses
        self.frequency_log_base = frequency_log_base

        # Precompute decay constant for exponential
        self.decay_constant = math.log(2) / self.recency_half_life_seconds

    def calculate_recency_score(
        self,
        created_at: datetime,
        current_time: Optional[datetime] = None
    ) -> float:
        """
        Calculate recency score using exponential decay.

        Formula: R = exp(-λ × t)
        Where λ = ln(2) / half_life

        Args:
            created_at: When memory was created (UTC)
            current_time: Current time (UTC), defaults to now
        Returns:
            Float 0.0 to 1.0 (1.0 = just created, 0.5 = 30 days old)
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        # Ensure timezone awareness
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)

        # Calculate time elapsed in seconds
        time_elapsed_seconds = (current_time - created_at).total_seconds()

        # Exponential decay: exp(-λ × t)
        recency_score = math.exp(-self.decay_constant * time_elapsed_seconds)

        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, recency_score))

    def calculate_frequency_score(
        self,
        access_count: int
    ) -> float:
        """
        Calculate frequency score using logarithmic scaling.

        Formula: F = log(count + 1) / log(max + 1)

        Justification: Diminishing returns (1st access more valuable than 100th)
        Args:
            access_count: Number of times memory accessed
        Returns:
            Float 0.0 to 1.0 (0.0 = never accessed, 1.0 = max accesses)
        """
        if access_count <= 0:
            return 0.0

        # Logarithmic scaling
        numerator = math.log(access_count + 1, self.frequency_log_base)
        denominator = math.log(self.frequency_max_accesses + 1, self.frequency_log_base)
        frequency_score = numerator / denominator

        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, frequency_score))

    def calculate_priority_score(
        self,
        recency_score: float,
        frequency_score: float,
        importance_score: float,
        recency_weight: float = 0.3,
        frequency_weight: float = 0.2,
        importance_weight: float = 0.5
    ) -> float:
        """
        Calculate combined priority score.

        Formula: P = (R × 0.3) + (F × 0.2) + (I × 0.5)
        Args:
            recency_score: 0.0 to 1.0
            frequency_score: 0.0 to 1.0
            importance_score: 0.0 to 1.0
            weights: Defaults from config

        Returns:
            Float 0.0 to 1.0
        """
        priority = (
            recency_score * recency_weight +
            frequency_score * frequency_weight +
            importance_score * importance_weight
        )

        return max(0.0, min(1.0, priority))
