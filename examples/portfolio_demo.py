"""Small executable tour of the installed pricing library; not a statistical study."""
from pathlib import Path
import json
from mc_pricer import (
    BlackScholesModel, EuropeanOption, ArithmeticAsianOption, HestonModel,
    make_rng, black_scholes_price, price_monte_carlo, price_heston_monte_carlo,
    finite_difference_greek, pathwise_delta,
)


def main() -> None:
    root=Path(__file__).resolve().parents[1]
    model=BlackScholesModel(100.,.05,.2)
    call=EuropeanOption(100.)
    vanilla=price_monte_carlo(model,call,1.,100_000,rng=make_rng(17))
    asian=price_monte_carlo(model,ArithmeticAsianOption(100.),1.,12_000,n_steps=12,
        variance_reduction='control_variate',n_pilot_paths=1200,rng=make_rng(18))
    fd=finite_difference_greek(model,call,1.,6000,greek='delta',bump=1.,rng=make_rng(19))
    pw=pathwise_delta(model,call,1.,12_000,rng=make_rng(20))
    heston=HestonModel.from_json((root/'examples/heston_calibration.json').read_text(),spot=100.,rate=.05)
    exotic=price_heston_monte_carlo(heston,ArithmeticAsianOption(100.),1.,12_000,384,
        n_observations=12,rng=make_rng(21))
    print(json.dumps(dict(
        european_call=dict(price=vanilla.price,se=vanilla.standard_error,
                           analytical=black_scholes_price(model,100.,1.)),
        asian_control=dict(price=asian.price,se=asian.standard_error,total_paths=asian.n_paths,
                           production_samples=asian.n_independent_samples,pilot=asian.n_pilot_paths),
        fd_delta=dict(estimate=fd.estimate,se=fd.standard_error,terminal_evaluations=fd.n_payoff_evaluations),
        pathwise_delta=dict(estimate=pw.estimate,se=pw.standard_error,terminal_evaluations=pw.n_terminal_evaluations),
        heston_asian=dict(price=exotic.price,se=exotic.standard_error,integration_steps=exotic.n_steps,
                          monitoring_dates=exotic.n_observations),
        note='Single seeded examples. Use the saved repeated experiments for accuracy/efficiency conclusions.'
    ),indent=2))


if __name__=='__main__':main()
