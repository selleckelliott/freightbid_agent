from ports.toll_estimator import TollEstimatorPort

DEFAULT_TOLL_STATES = {
    "NY": 0.18,
    "NJ": 0.18,
    "PA": 0.15,
    "IL": 0.10,
    "OH": 0.10,
    "IN": 0.10,
    "FL": 0.12,
    "OK": 0.08,
    "KS": 0.08,
    "WV": 0.10,
    "MA": 0.12,
}


class FlatRateTollEstimator(TollEstimatorPort):
    def __init__(self, per_mile_default: float = 0.0, per_state: dict | None = None):
        self.per_mile_default = per_mile_default
        self.per_state = per_state if per_state is not None else DEFAULT_TOLL_STATES

    def estimate(self, miles: float, origin_state: str, destination_state: str) -> float:
        if miles <= 0:
            return 0.0
        o = self.per_state.get((origin_state or "").upper(), self.per_mile_default)
        d = self.per_state.get((destination_state or "").upper(), self.per_mile_default)
        avg = (o + d) / 2.0
        return miles * avg
