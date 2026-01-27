import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

warnings.filterwarnings('ignore')


@dataclass
class ScenarioConfig:
    """Configuration for distress/default scenarios"""
    member_pct: float
    installments: Tuple[int, int]


class SukukEnhancementFund:
    """
    Sukuk Enhancement Fund Simulation Model (Static N=30)
    """

    # Class constants
    MONTHS_PER_YEAR = 12
    PRINCIPAL_MIN = 100
    PRINCIPAL_MAX = 1000
    PREMIUM_MIN = 0.0
    PREMIUM_MAX = 0.10
    DEFAULT_N_MEMBERS = 30
    DEFAULT_MATURITY_YEARS = 5
    DEFAULT_BASE_RATE = 0.01
    DEFAULT_DELTA = 0.5
    DEFAULT_ALPHA_PRIME = 0.5
    DEFAULT_BETA = 0.5

    # Scenario configurations
    DISTRESS_SCENARIOS = {
        'optimistic': ScenarioConfig(member_pct=0.05, installments=(1, 3)),
        'realistic': ScenarioConfig(member_pct=0.09, installments=(4, 6)),
        'pessimistic': ScenarioConfig(member_pct=0.20, installments=(7, 10))
    }

    DEFAULT_SCENARIOS = {
        'optimistic': ScenarioConfig(member_pct=0.05, installments=(30, 36)),
        'realistic': ScenarioConfig(member_pct=0.09, installments=(30, 36)),
        'pessimistic': ScenarioConfig(member_pct=0.20, installments=(30, 36))
    }

    def __init__(
            self,
            n_members: int = DEFAULT_N_MEMBERS,
            maturity_years: int = DEFAULT_MATURITY_YEARS,
            base_rate: float = DEFAULT_BASE_RATE,
            delta: float = DEFAULT_DELTA,
            alpha_prime: float = DEFAULT_ALPHA_PRIME,
            beta: float = DEFAULT_BETA
    ) -> None:
        self.n_members = n_members
        self.maturity_years = maturity_years
        self.months = maturity_years * self.MONTHS_PER_YEAR
        self.base_rate = base_rate
        self.delta = delta
        self.alpha_prime = alpha_prime
        self.beta = beta

        self.theoretical_params = self._calculate_theoretical_parameters()

    def _calculate_theoretical_parameters(self) -> Dict[str, float]:
        """Calculate theoretical distribution parameters for robust baselines."""
        theo_mean_principal = (self.PRINCIPAL_MIN + self.PRINCIPAL_MAX) / 2
        theo_mean_premium = (self.PREMIUM_MIN + self.PREMIUM_MAX) / 2
        theo_mean_rate_monthly = (self.base_rate + theo_mean_premium) / self.MONTHS_PER_YEAR
        theo_max_premium = self.PREMIUM_MAX
        theo_max_rate_monthly = (self.base_rate + theo_max_premium) / self.MONTHS_PER_YEAR

        theo_mean_debt = theo_mean_principal * (1 + self.months * theo_mean_rate_monthly)
        theo_max_debt = self.PRINCIPAL_MAX * (1 + self.months * theo_max_rate_monthly)

        return {
            'mean_principal': theo_mean_principal,
            'mean_installment': theo_mean_debt / self.months,
            'max_installment': theo_max_debt / self.months
        }

    def generate_sample(self, seed: Optional[int] = None) -> Dict[str, float]:
        if seed is not None:
            np.random.seed(seed)

        self.principals = np.random.uniform(self.PRINCIPAL_MIN, self.PRINCIPAL_MAX, self.n_members)
        self.market_premiums = np.random.uniform(self.PREMIUM_MIN, self.PREMIUM_MAX, self.n_members)

        monthly_base_rate = self.base_rate / self.MONTHS_PER_YEAR
        monthly_premiums_rate = self.market_premiums / self.MONTHS_PER_YEAR

        self.total_debts = self.principals * (1 + self.months * (monthly_base_rate + monthly_premiums_rate))
        self.monthly_installments = self.total_debts / self.months

        monthly_market_premiums = self.principals * monthly_premiums_rate
        self.K1_monthly = np.sum(monthly_market_premiums)

        self.K2_gross_monthly = np.sum(self.monthly_installments)
        self.K2_net_monthly = self.K2_gross_monthly - (self.delta * self.K1_monthly)

        self.K1_annual = self.K1_monthly * self.MONTHS_PER_YEAR
        self.K2_net_annual = self.K2_net_monthly * self.MONTHS_PER_YEAR

        self.alpha = (self.delta * self.K1_monthly) / (self.alpha_prime * self.K2_net_monthly)

        return {
            'K1_annual': self.K1_annual,
            'K2_net_annual': self.K2_net_annual,
            'alpha': self.alpha
        }

    def simulate_fund_performance(self, n_iterations=10000, seed_offset=0) -> Dict[str, List]:
        results = {
            'annual_revenues': [],
            'annual_reserves': [],
            'cumulative_reserves': [],
            'augmented_alpha': [],
            'dividends': []
        }

        for i in range(n_iterations):
            sample_stats = self.generate_sample(seed=i + seed_offset)
            annual_revenue = self.delta * sample_stats['K1_annual']
            annual_reserve_contribution = self.beta * annual_revenue

            iter_cum_reserves = np.zeros(self.maturity_years)
            iter_aug_alpha = np.zeros(self.maturity_years)
            iter_dividends = np.zeros(self.maturity_years)

            exposure = self.alpha_prime * sample_stats['K2_net_annual']
            cum_reserves_accumulated = 0.0

            for year in range(self.maturity_years):
                capacity = annual_revenue + cum_reserves_accumulated

                if exposure == 0:
                    iter_aug_alpha[year] = 0.0
                else:
                    iter_aug_alpha[year] = capacity / exposure

                iter_dividends[year] = (1 - self.beta) * annual_revenue
                cum_reserves_accumulated += annual_reserve_contribution
                iter_cum_reserves[year] = cum_reserves_accumulated

            results['annual_revenues'].append(annual_revenue)
            results['annual_reserves'].append(annual_reserve_contribution)
            results['cumulative_reserves'].append(iter_cum_reserves)
            results['augmented_alpha'].append(iter_aug_alpha)
            results['dividends'].append(iter_dividends)

        return results

    def analyze_scenarios(self, results, capital_injection=0, event_year=3):
        """
        Analyze scenarios returning detailed coverage data.
        Capacity is FIXED (Reserves + Year 3 Revenue) for all scenarios.
        Drawdown calculation uses this fixed capacity.
        """

        avg_annual_revenue = np.mean(results['annual_revenues'])
        avg_cumulative_reserves = np.mean(results['cumulative_reserves'], axis=0)

        # Baseline Capacity (Year 3 Event)
        prior_year_idx = event_year - 2
        reserves_prior = avg_cumulative_reserves[prior_year_idx] if prior_year_idx >= 0 else 0

        # CONSISTENT CAPACITY DEFINITION
        base_internal_capacity = reserves_prior + avg_annual_revenue
        base_enhanced_capacity = base_internal_capacity + capital_injection

        # Installment Stats
        avg_inst = self.theoretical_params['mean_installment']
        max_inst = self.theoretical_params['max_installment']
        distressed_installment = (avg_inst + max_inst) / 2

        final_output = {'distress': {}, 'default': {}}

        def process(config_dict, key):
            is_default = (key == 'default')

            for name, config in config_dict.items():
                n_members = config.member_pct * self.n_members
                n_installments = np.mean(config.installments)

                total_losses = n_members * n_installments * distressed_installment
                fund_exposure = self.alpha_prime * total_losses

                # --- DRAWDOWN LOGIC ---
                if is_default:
                    # Drawdown is the amount of EXPOSURE that Internal Capacity CANNOT cover.
                    deficit = max(0, fund_exposure - base_internal_capacity)

                    # We assume Capital covers the deficit up to its limit
                    drawdown = min(deficit, capital_injection)

                    if capital_injection > 0:
                        drawdown_pct = drawdown / capital_injection
                    else:
                        drawdown_pct = 0.0
                else:
                    drawdown = 0.0
                    drawdown_pct = 0.0

                # Coverage Calculation
                if fund_exposure == 0:
                    cov_int = float('inf')
                    cov_enh = float('inf')
                else:
                    cov_int = base_internal_capacity / fund_exposure
                    cov_enh = base_enhanced_capacity / fund_exposure

                final_output[key][name] = {
                    'n_members': n_members,
                    'total_losses': total_losses,
                    'fund_exposure': fund_exposure,
                    'internal_capacity': base_internal_capacity,  # Constant
                    'cov_internal': cov_int,
                    'cov_enhanced': cov_enh,
                    'drawdown': drawdown,
                    'drawdown_pct': drawdown_pct
                }

        process(self.DISTRESS_SCENARIOS, 'distress')
        process(self.DEFAULT_SCENARIOS, 'default')

        return final_output, distressed_installment

    def parameter_sensitivity_analysis(self, n_iterations=2000):
        param_ranges = {
            'delta': np.linspace(0.3, 0.7, 5),
            'beta': np.linspace(0.3, 0.7, 5),
            'alpha_prime': np.linspace(0.3, 0.7, 5)
        }
        sensitivity_results = {}
        for param_name, values in param_ranges.items():
            print(f"  ...Analyzing {param_name}...")
            sensitivity_results[param_name] = []
            original_value = getattr(self, param_name)
            for value in values:
                setattr(self, param_name, value)
                results = self.simulate_fund_performance(n_iterations=n_iterations, seed_offset=999)
                avg_alpha_y5 = np.mean([alpha[4] for alpha in results['augmented_alpha']])
                avg_rev = np.mean(results['annual_revenues'])
                sensitivity_results[param_name].append({
                    'value': value,
                    'augmented_alpha_y5': avg_alpha_y5,
                    'annual_revenue': avg_rev
                })
            setattr(self, param_name, original_value)
        return sensitivity_results

    def calculate_risk_metrics(self, results):
        revenues = np.array(results['annual_revenues'])
        var_95_rev = np.percentile(revenues, 5)
        alpha_y5 = np.array([alpha[4] for alpha in results['augmented_alpha']])
        var_95_alpha = np.percentile(alpha_y5, 5)

        return {
            'var_95_revenue': var_95_rev,
            'var_95_alpha': var_95_alpha,
            'alpha_volatility': np.std(alpha_y5),
            'avg_alpha_y5': np.mean(alpha_y5)
        }


def create_visualizations(
        results: Dict[str, List],
        sens_results: Dict[str, List[Dict[str, float]]],
        scenario_data: Dict[str, Dict[str, Dict[str, float]]],
        maturity_years: int
) -> None:
    """
    Create comprehensive visualization dashboard (Static Simulation).
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    years = range(1, maturity_years + 1)
    avg_cumulative_reserves = np.mean(results['cumulative_reserves'], axis=0)
    avg_augmented_alpha = np.mean(results['augmented_alpha'], axis=0)

    # Plot A: Cumulative Reserves Growth
    axes[0, 0].plot(years, avg_cumulative_reserves, 'b-o', linewidth=2, markersize=8)
    axes[0, 0].set_title('Cumulative Reserves Growth Over Time', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Year', fontsize=10)
    axes[0, 0].set_ylabel('USD', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(years)

    # Add labels
    for year, value in zip(years, avg_cumulative_reserves):
        axes[0, 0].annotate(f'${value:.0f}', (year, value), xytext=(0, 10),
                            textcoords="offset points", ha='center', fontsize=8)

    # Plot B: Augmented Alpha with CI
    alpha_all = np.array(results['augmented_alpha'])
    lower = np.percentile(alpha_all, 5, axis=0)
    upper = np.percentile(alpha_all, 95, axis=0)

    axes[0, 1].plot(years, avg_augmented_alpha, 'g-o', label='Mean Alpha', linewidth=2)
    axes[0, 1].fill_between(years, lower, upper, color='green', alpha=0.1, label='90% CI')
    axes[0, 1].axhline(y=0.5, color='orange', linestyle='--', label='50% Target')
    axes[0, 1].set_title('Augmented Alpha Progression', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Alpha')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xticks(years)

    # Plot C: Sensitivity (Delta)
    deltas = [x['value'] for x in sens_results['delta']]
    alphas = [x['augmented_alpha_y5'] for x in sens_results['delta']]
    axes[1, 0].plot(deltas, alphas, 'm-o', linewidth=2)
    axes[1, 0].set_title('Sensitivity: Delta vs Year 5 Alpha', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Delta')
    axes[1, 0].set_ylabel('Year 5 Alpha')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.3)

    # Plot D: Coverage Ratio Bar Chart (Enhanced)
    scenarios = list(scenario_data['distress'].keys())
    # Use Enhanced Coverage for visual impact
    distress_cov = [scenario_data['distress'][s]['cov_enhanced'] for s in scenarios]
    default_cov = [scenario_data['default'][s]['cov_enhanced'] for s in scenarios]

    x = np.arange(len(scenarios))
    width = 0.35

    bars1 = axes[1, 1].bar(x - width / 2, distress_cov, width, label='Distress', color='steelblue')
    bars2 = axes[1, 1].bar(x + width / 2, default_cov, width, label='Default', color='darkorange')

    axes[1, 1].axhline(y=1.0, color='red', linestyle='--', label='Breakeven')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([s.capitalize() for s in scenarios])
    axes[1, 1].set_title('Enhanced Coverage Ratio (With Capital)', fontsize=12, fontweight='bold')
    axes[1, 1].legend()

    # Add labels
    for i, v in enumerate(distress_cov):
        axes[1, 1].text(i - width / 2, v + 0.1, f'{v:.1f}x', ha='center', fontsize=8)
    for i, v in enumerate(default_cov):
        axes[1, 1].text(i + width / 2, v + 0.1, f'{v:.1f}x', ha='center', fontsize=8)

    plt.tight_layout()
    plt.show()


def run_full_analysis(n_members=30, maturity_years=5, n_iterations=10000, verbose=True):
    if verbose:
        print("=" * 100)
        print("SUKUK ENHANCEMENT FUND: STATIC ANALYSIS (N=30)")
        print("=" * 100)

    sef = SukukEnhancementFund(n_members=n_members, maturity_years=maturity_years)

    # 1. INSTALLMENT STATISTICS
    params = sef.theoretical_params
    avg_inst = params['mean_installment']
    max_inst = params['max_installment']
    distressed_inst = (avg_inst + max_inst) / 2

    if verbose:
        print("\nINSTALLMENT STATISTICS (Baseline vs. Stress)")
        print("-" * 60)
        print(f"{'Metric':<40} | {'Value':>10}")
        print("-" * 60)
        print(f"{'Mean Installment (Whole Sample)':<40} | ${avg_inst:>9.2f}")
        print(f"{'Maximum Installment (Whole Sample)':<40} | ${max_inst:>9.2f}")
        print(f"{'Average of Upper 50% (Distressed)':<40} | ${distressed_inst:>9.2f}")
        print("-" * 60)

    # 2. Run Simulation
    if verbose: print(f"\nRunning Simulation ({n_iterations:,} iterations)...")
    results = sef.simulate_fund_performance(n_iterations=n_iterations)

    # 3. Financial Projections
    avg_annual_revenue = np.mean(results['annual_revenues'])
    avg_cumulative_reserves = np.mean(results['cumulative_reserves'], axis=0)
    avg_augmented_alpha = np.mean(results['augmented_alpha'], axis=0)

    if verbose:
        print("\n5-YEAR FUND PROJECTIONS (Simulation Averages)")
        print("-" * 75)
        print(f"{'Year':<6} | {'Annual Rev':<12} | {'Cum Reserves':<15} | {'Aug Alpha':<10}")
        print("-" * 75)
        for year in range(maturity_years):
            print(
                f"{year + 1:<6} | ${avg_annual_revenue:<11.2f} | ${avg_cumulative_reserves[year]:<14.2f} | {avg_augmented_alpha[year]:.2%}")

    # 4. Capital & Dividend Analysis
    capital_req = avg_annual_revenue * 4

    if verbose:
        print("\nCAPITAL & DIVIDEND ANALYSIS")
        print("-" * 60)
        print(f"Required Capital (4 Years Upfront):  ${capital_req:,.2f}")

    # 5. Scenario Analysis
    scenarios, _ = sef.analyze_scenarios(results, capital_injection=capital_req, event_year=3)

    if verbose:
        print(f"\nSCENARIO ANALYSIS (Year 3 Event) - Capital: ${capital_req:,.0f}")

        # --- DISTRESS TABLE (Without Drawdown) ---
        print("\nDISTRESS SCENARIOS:")
        headers_dist = f"{'SCENARIO':<12} | {'AFFECTED':<8} | {'TOTAL LOSS':<12} | {'FUND EXP':<10} | {'CAPACITY':<10} | {'INT. COV':<10} | {'ENH. COV':<10}"
        print("-" * 95)
        print(headers_dist)
        print("-" * 95)
        for name, res in scenarios['distress'].items():
            loss = f"${res['total_losses']:,.0f}"
            exp = f"${res['fund_exposure']:,.0f}"
            cap = f"${res['internal_capacity']:,.0f}"
            icov = f"{res['cov_internal']:.2f}x"
            ecov = f"{res['cov_enhanced']:.2f}x"
            print(
                f"  {name.capitalize():<10} | {res['n_members']:<8.1f} | {loss:<12} | {exp:<10} | {cap:<10} | {icov:<10} | {ecov:<10}")

        # --- DEFAULT TABLE (With Drawdown) ---
        print("\nDEFAULT SCENARIOS (Capacity unchanged):")
        headers_def = f"{'SCENARIO':<12} | {'LOSS':<10} | {'FUND EXP':<10} | {'CAPACITY':<10} | {'INT. COV':<9} | {'ENH. COV':<9} | {'DRAWDOWN':<10} | {'% CAP'}"
        print("-" * 110)
        print(headers_def)
        print("-" * 110)
        for name, res in scenarios['default'].items():
            loss = f"${res['total_losses']:,.0f}"
            exp = f"${res['fund_exposure']:,.0f}"
            cap = f"${res['internal_capacity']:,.0f}"
            icov = f"{res['cov_internal']:.2f}x"
            ecov = f"{res['cov_enhanced']:.2f}x"
            dd = f"${res['drawdown']:,.0f}"
            dd_pct = f"{res['drawdown_pct']:.1%}"
            print(
                f"  {name.capitalize():<10} | {loss:<10} | {exp:<10} | {cap:<10} | {icov:<9} | {ecov:<9} | {dd:<10} | {dd_pct}")
        print("-" * 110)

    # 6. Sensitivity
    if verbose:
        print("\nPARAMETER SENSITIVITY (Alpha Year 5)")
        print("-" * 60)
        sens = sef.parameter_sensitivity_analysis(n_iterations=2000)
        for p, table in sens.items():
            print(f"\nSensitivity: {p.upper()}")
            print(f"{'Value':<10} | {'Year 5 Alpha':<15} | {'Annual Rev':<15}")
            print("-" * 45)
            for row in table:
                print(f"{row['value']:<10.2f} | {row['augmented_alpha_y5']:<15.2%} | ${row['annual_revenue']:<14.2f}")

    # 6b. Risk Metrics Printout
    if verbose:
        print("\nRISK METRICS (Year 5 Projection)")
        print("-" * 60)
        # Calculate percentiles directly from the alpha array for Year 5
        alpha_y5 = np.array([a[4] for a in results['augmented_alpha']])

        mean_alpha = np.mean(alpha_y5)
        std_alpha = np.std(alpha_y5)
        var_95 = np.percentile(alpha_y5, 5)  # 5th percentile = 95% confidence
        var_99 = np.percentile(alpha_y5, 1)  # 1st percentile = 99% confidence

        print(f"Mean Alpha:           {mean_alpha:.2%}")
        print(f"Volatility (Std Dev): {std_alpha:.2%}")
        print(f"VaR (95% Confidence): {var_95:.2%} (The 'Solvency Floor')")
        print(f"VaR (99% Confidence): {var_99:.2%}")


    # 7. Plots
    if verbose:
        print("\nGenerating Visualizations...")
        create_visualizations(results, sens, scenarios, maturity_years)

    return sef, results


if __name__ == "__main__":
    run_full_analysis()
