"""Risk-neutral Heston parameters and explicit calibration handoff."""
from collections.abc import Mapping
from dataclasses import dataclass, fields
import json
from mc_pricer._validation import finite_real


@dataclass(frozen=True)
class HestonModel:
    """Non-dividend-paying Heston model; time in years, variance in year^-1.

    v0 and theta are VARIANCES, not volatilities. kappa>0, theta/v0/xi>=0,
    rho in [-1,1]. Feller failure is allowed, not silently repaired. Parameters
    are interpreted under Q; no physical-to-risk-neutral conversion/calibration.
    """
    spot: float
    rate: float
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float

    def __post_init__(self) -> None:
        for field in fields(self):
            object.__setattr__(self, field.name, finite_real(field.name, getattr(self, field.name)))
        if self.spot <= 0 or self.kappa <= 0:
            raise ValueError('spot and kappa must be positive')
        if min(self.v0, self.theta, self.xi) < 0:
            raise ValueError('v0, theta and xi must be nonnegative')
        if not -1 <= self.rho <= 1:
            raise ValueError('rho must be between -1 and 1')

    @property
    def feller_margin(self) -> float:
        """2*kappa*theta-xi², a diagnostic, not an Euler stability guarantee."""
        return 2*self.kappa*self.theta-self.xi**2

    @classmethod
    def from_dict(cls, parameters: Mapping[str, float], *, spot: float | None = None,
                  rate: float | None = None) -> 'HestonModel':
        """Accept a flat exact-key mapping; allow separate spot/rate inputs.

        Pass a nested calibration result's parameter submapping explicitly.
        Unknown/missing keys and duplicate sources are rejected to catch typos.
        No renaming, clipping, unit guessing or Feller enforcement occurs.
        """
        if not isinstance(parameters, Mapping):
            raise TypeError('parameters must be a mapping')
        data = dict(parameters)
        for key, value in (('spot', spot), ('rate', rate)):
            if value is not None:
                if key in data:
                    raise ValueError(f'{key} supplied both in mapping and as argument')
                data[key] = value
        expected = {f.name for f in fields(cls)}
        if set(data) != expected:
            raise ValueError(f'parameter keys must be {sorted(expected)}; missing={expected-set(data)}, unexpected={set(data)-expected}')
        return cls(**data)

    @classmethod
    def from_json(cls, text: str, *, spot: float | None = None,
                  rate: float | None = None) -> 'HestonModel':
        """Parse JSON TEXT, not a filename, with duplicate-key rejection.

        For a file, call Path(filename).read_text() explicitly. A calibration
        wrapper with metadata must first be parsed and its parameters selected.
        """
        if not isinstance(text, str):
            raise TypeError('text must be a JSON string')
        def unique_pairs(pairs):
            data = {}
            for key, value in pairs:
                if key in data:
                    raise ValueError(f'duplicate JSON key: {key}')
                data[key] = value
            return data
        return cls.from_dict(json.loads(text, object_pairs_hook=unique_pairs), spot=spot, rate=rate)
