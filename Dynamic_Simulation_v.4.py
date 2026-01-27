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
    member_pct: float
    installments: Tuple[int, int]


class DynamicSukukFund:
    """
    Dynamic Sukuk Enhancement Fund Model (10-Year Growth & Turnover)
    """

    # Class Constants
    MONTHS_PER_YEAR = 12
    PRINCIPAL_MIN = 100
    PRINCIPAL_MAX = 1000
    PREMIUM_MIN = 0.0
    PREMIUM_MAX = 0.10

    # Defaults
    DEFAULT_MATURITY_YEARS = 5
    DEFAULT_BASE_RATE = 0.01
    DEFAULT_DELTA = 0.5
    DEFAULT_ALPHA_PRIME = 0.5
    DEFAULT_BETA = 0.5

    # Scenario Configs
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

    def __init__(self, base_rate=DEFAULT_BASE_RATE, delta=DEFAULT_DELTA,
                 alpha_prime=DEFAULT_ALPHA_PRIME, beta=DEFAULT_BETA):
        self.base_rate = base_rate
        self.delta = delta
        self.alpha_prime = alpha_prime
        self.beta = beta

        # Cohort Schedule: Start 30 -> Add 10/yr -> Year 6 Turnover
        self.cohort_schedule = [
            (1, 30),
            (2, 10), (3, 10), (4, 10), (5, 10),
            (6, 30),
            (7, 10), (8, 10), (9, 10), (10, 10)
        ]
        self.simulation_horizon = 10
        self.theoretical_params = self._calculate_theoretical_parameters()

    def _calculate_theoretical_parameters(self):
        """Calculate robust baselines for installment sizes"""
        theo_mean_principal = (self.PRINCIPAL_MIN + self.PRINCIPAL_MAX) / 2
        theo_mean_premium = (self.PREMIUM_MIN + self.PREMIUM_MAX) / 2
        theo_mean_rate_monthly = (self.base_rate + theo_mean_premium) / 12
        theo_max_premium = self.PREMIUM_MAX
        theo_max_rate_monthly = (self.base_rate + theo_max_premium) / 12

        theo_mean_debt = theo_mean_principal * (1 + (5 * 12) * theo_mean_rate_monthly)
        theo_max_debt = self.PRINCIPAL_MAX * (1 + (5 * 12) * theo_max_rate_monthly)

        return {
            'mean_installment': theo_mean_debt / (5 * 12),
            'max_installment': theo_max_debt / (5 * 12)
        }

    def generate_cohort(self, n_members, seed=None):
        if seed is not None: np.random.seed(seed)
        principals = np.random.uniform(self.PRINCIPAL_MIN, self.PRINCIPAL_MAX, n_members)
        premiums = np.random.uniform(self.PREMIUM_MIN, self.PREMIUM_MAX, n_members)

        monthly_base = self.base_rate / 12
        monthly_prem = premiums / 12
        months = 60  # 5 years

        total_debt = principals * (1 + months * (monthly_base + monthly_prem))
        monthly_installments = total_debt / months

        k1_annual = np.sum(principals * monthly_prem) * 12
        k2_gross_monthly = np.sum(monthly_installments)
        k2_net_monthly = k2_gross_monthly - (self.delta * (k1_annual / 12))
        k2_net_annual = k2_net_monthly * 12

        return {'revenue': k1_annual, 'exposure': k2_net_annual}

    def simulate_dynamic_performance(self, n_iterations=10000, seed_offset=0):
        results = {
            'total_revenues': [], 'total_reserves': [],
            'augmented_alpha': [], 'active_members': [],
            'cumulative_reserves_history': [],
            'total_dividends': [],
            'total_exposure': []
        }

        for i in range(n_iterations):
            cohorts = []
            for start_year, size in self.cohort_schedule:
                c_seed = i + seed_offset + (start_year * 1000)
                stats = self.generate_cohort(size, seed=c_seed)
                cohorts.append({
                    'start': start_year,
                    'end': start_year + 4,
                    'revenue': self.delta * stats['revenue'],
                    'exposure': self.alpha_prime * stats['exposure'],
                    'size': size
                })

            iter_alpha = np.zeros(self.simulation_horizon)
            iter_cum_reserves = np.zeros(self.simulation_horizon)
            iter_active_members = np.zeros(self.simulation_horizon)
            iter_total_rev = np.zeros(self.simulation_horizon)
            iter_dividends = np.zeros(self.simulation_horizon)
            iter_total_exp = np.zeros(self.simulation_horizon)

            cum_reserves = 0.0

            for t in range(1, self.simulation_horizon + 1):
                active_rev = 0.0
                active_exp = 0.0
                active_count = 0

                for c in cohorts:
                    if c['start'] <= t <= c['end']:
                        active_rev += c['revenue']
                        active_exp += c['exposure']
                        active_count += c['size']

                capacity = active_rev + cum_reserves
                if active_exp > 0:
                    iter_alpha[t - 1] = capacity / active_exp

                iter_dividends[t - 1] = (1 - self.beta) * active_rev
                cum_reserves += (self.beta * active_rev)

                iter_cum_reserves[t - 1] = cum_reserves
                iter_active_members[t - 1] = active_count
                iter_total_rev[t - 1] = active_rev
                iter_total_exp[t - 1] = active_exp

            results['total_revenues'].append(iter_total_rev)
            results['cumulative_reserves_history'].append(iter_cum_reserves)
            results['augmented_alpha'].append(iter_alpha)
            results['active_members'].append(iter_active_members)
            results['total_dividends'].append(iter_dividends)
            results['total_exposure'].append(iter_total_exp)

        return results

    def analyze_scenarios(self, results, capital_injection=0, event_year=5):
        """Analyze scenarios at a specific event year."""
        avg_rev_history = np.mean(results['total_revenues'], axis=0)
        avg_res_history = np.mean(results['cumulative_reserves_history'], axis=0)
        avg_members_history = np.mean(results['active_members'], axis=0)

        revenue_event_year = avg_rev_history[event_year - 1]
        reserves_prior_year = avg_res_history[event_year - 2] if event_year > 1 else 0

        base_internal_capacity = reserves_prior_year + revenue_event_year
        base_enhanced_capacity = base_internal_capacity + capital_injection

        active_members_count = avg_members_history[event_year - 1]

        avg_inst = self.theoretical_params['mean_installment']
        max_inst = self.theoretical_params['max_installment']
        distressed_installment = (avg_inst + max_inst) / 2

        final_output = {'distress': {}, 'default': {}}

        def process(config_dict, key):
            is_default = (key == 'default')
            for name, config in config_dict.items():
                n_members = config.member_pct * active_members_count
                n_installments = np.mean(config.installments)
                total_losses = n_members * n_installments * distressed_installment
                fund_exposure = self.alpha_prime * total_losses

                if is_default:
                    deficit = max(0, fund_exposure - base_internal_capacity)
                    drawdown = min(deficit, capital_injection)
                    drawdown_pct = drawdown / capital_injection if capital_injection > 0 else 0.0
                else:
                    drawdown = 0.0
                    drawdown_pct = 0.0

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
                    'internal_capacity': base_internal_capacity,
                    'cov_internal': cov_int,
                    'cov_enhanced': cov_enh,
                    'drawdown': drawdown,
                    'drawdown_pct': drawdown_pct
                }

        process(self.DISTRESS_SCENARIOS, 'distress')
        process(self.DEFAULT_SCENARIOS, 'default')

        return final_output, revenue_event_year

    def parameter_sensitivity_analysis(self, n_iterations=2000):
        param_ranges = {'delta': np.linspace(0.3, 0.7, 5)}
        sensitivity_results = {}
        for param_name, values in param_ranges.items():
            print(f"  ...Analyzing {param_name}...")
            sensitivity_results[param_name] = []
            original_value = getattr(self, param_name)
            for value in values:
                setattr(self, param_name, value)
                results = self.simulate_dynamic_performance(n_iterations=n_iterations, seed_offset=999)
                avg_alpha_y5 = np.mean([a[4] for a in results['augmented_alpha']])
                avg_alpha_y10 = np.mean([a[9] for a in results['augmented_alpha']])
                sensitivity_results[param_name].append({
                    'value': value, 'alpha_y5': avg_alpha_y5, 'alpha_y10': avg_alpha_y10
                })
            setattr(self, param_name, original_value)
        return sensitivity_results


def create_visualizations(results, sens_results, maturity_years):
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    years = range(1, maturity_years + 1)

    avg_alpha = np.mean(results['augmented_alpha'], axis=0)
    avg_reserves = np.mean(results['cumulative_reserves_history'], axis=0)
    avg_members = np.mean(results['active_members'], axis=0)
    avg_exposure = np.mean(results['total_exposure'], axis=0)

    # Calculate Uncovered Exposure for Graph
    uncovered_exposure = avg_exposure * (1 - avg_alpha)

    # Plot A: Alpha
    alpha_all = np.array(results['augmented_alpha'])
    lower = np.percentile(alpha_all, 5, axis=0)
    upper = np.percentile(alpha_all, 95, axis=0)

    axes[0, 0].plot(years, avg_alpha, 'g-o', linewidth=2, label='Mean Alpha')
    axes[0, 0].fill_between(years, lower, upper, color='green', alpha=0.1, label='90% CI')
    axes[0, 0].axhline(y=0.5, color='red', linestyle='--', label='Target 50%')
    axes[0, 0].set_title('Protection Capacity (Augmented Alpha)')
    axes[0, 0].set_ylabel('Augmented Alpha')
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_xticks(years)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot B: Members vs Reserves
    ax1 = axes[0, 1]
    ax2 = ax1.twinx()
    ax1.bar(years, avg_members, alpha=0.3, color='blue', label='Members (Left)')
    ax2.plot(years, avg_reserves, 'r-o', linewidth=2, label='Reserves (Right)')
    ax1.set_ylabel('Members', color='blue')
    ax2.set_ylabel('Reserves ($)', color='red')
    axes[0, 1].set_title('Fund Growth: Members vs. Reserves')
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc='upper left')

    # Plot C: Sensitivity (Delta)
    deltas = [x['value'] for x in sens_results['delta']]
    a5 = [x['alpha_y5'] for x in sens_results['delta']]
    a10 = [x['alpha_y10'] for x in sens_results['delta']]

    axes[1, 0].plot(deltas, a5, 'o--', color='orange', label='Year 5 (Weakest)')
    axes[1, 0].plot(deltas, a10, 'o-', color='purple', label='Year 10 (Steady)')
    axes[1, 0].set_title('Sensitivity: Delta vs Alpha')
    axes[1, 0].set_xlabel('Delta')
    axes[1, 0].set_ylabel('Alpha')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot D: Exposure vs Uncovered Exposure (Gap Analysis)
    axes[1, 1].plot(years, avg_exposure, 'k-o', linewidth=2, label='Total Exposure')
    axes[1, 1].plot(years, uncovered_exposure, 'r-o', linewidth=2, label='Uncovered Exposure (Gap)')
    axes[1, 1].axhline(y=0, color='gray', linestyle='--', linewidth=1)

    axes[1, 1].fill_between(years, uncovered_exposure, 0, where=(uncovered_exposure > 0), color='red', alpha=0.1,
                            label='Deficit Zone')
    axes[1, 1].fill_between(years, uncovered_exposure, 0, where=(uncovered_exposure < 0), color='green', alpha=0.1,
                            label='Surplus Zone')

    axes[1, 1].set_title('Risk Gap Analysis: Exposure vs. Uncovered Amount')
    axes[1, 1].set_ylabel('USD Amount ($)')
    axes[1, 1].set_xlabel('Year')
    axes[1, 1].set_xticks(years)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def run_dynamic_analysis(n_iterations=10000, verbose=True):
    if verbose:
        print("=" * 100)
        print("SUKUK ENHANCEMENT FUND: DYNAMIC ANALYSIS (10 YEARS)")
        print("=" * 100)

    fund = DynamicSukukFund()
    if verbose: print(f"\nRunning Simulation ({n_iterations:,} iterations)...")
    results = fund.simulate_dynamic_performance(n_iterations=n_iterations)

    # Financials
    avg_alpha = np.mean(results['augmented_alpha'], axis=0)
    avg_reserves = np.mean(results['cumulative_reserves_history'], axis=0)
    avg_members = np.mean(results['active_members'], axis=0)
    avg_revenue = np.mean(results['total_revenues'], axis=0)
    avg_exposure = np.mean(results['total_exposure'], axis=0)
    avg_dividends = np.mean(results['total_dividends'], axis=0)  # ADDED

    # Uncovered Exposure
    uncovered_exposure = avg_exposure * (1 - avg_alpha)

    # Capital Calculation (Upfront)
    capital_req = avg_revenue[0] * 4

    if verbose:
        print("\n1. FINANCIAL PROJECTIONS (Includes Dividends)")
        print("-" * 105)
        # Added Dividends column to header and row
        print(
            f"{'Year':<5} | {'Members':<8} | {'Revenue':<12} | {'Dividends':<12} | {'Cum Reserves':<15} | {'Aug Alpha':<10}")
        print("-" * 105)
        for t in range(10):
            print(
                f"{t + 1:<5} | {avg_members[t]:<8.0f} | ${avg_revenue[t]:<11.2f} | ${avg_dividends[t]:<11.2f} | ${avg_reserves[t]:<14.2f} | {avg_alpha[t]:.2%}")

    if verbose:
        print("\n2. RISK GAP ANALYSIS (Exposure vs Uncovered)")
        print("-" * 80)
        print(f"{'Year':<5} | {'Total Exposure':<15} | {'Uncovered Exposure (Gap)':<25}")
        print("-" * 80)
        for t in range(10):
            gap_str = f"${uncovered_exposure[t]:,.1f}"
            if uncovered_exposure[t] < 0:
                gap_str += " (Surplus)"
            print(f"{t + 1:<5} | ${avg_exposure[t]:<14,.1f} | {gap_str:<25}")

    if verbose:
        print("\n3. CAPITAL ANALYSIS")
        print("-" * 60)
        print(f"External Capital Available (Fixed Upfront): ${capital_req:,.2f}")

    # Scenarios (Year 5)
    scenarios, _ = fund.analyze_scenarios(results, capital_injection=capital_req, event_year=5)

    if verbose:
        print(f"\n4. SCENARIO ANALYSIS (Year 5 - The 'Pinch Point')")

        # DISTRESS
        print("\nDISTRESS SCENARIOS:")
        headers_dist = f"{'SCENARIO':<12} | {'LOSS':<12} | {'FUND EXP':<10} | {'CAPACITY':<10} | {'INT. COV':<10} | {'ENH. COV':<10}"
        print("-" * 95)
        print(headers_dist)
        print("-" * 95)
        for name, res in scenarios['distress'].items():
            loss = f"${res['total_losses']:,.0f}"
            exp = f"${res['fund_exposure']:,.0f}"
            cap = f"${res['internal_capacity']:,.0f}"
            icov = f"{res['cov_internal']:.2f}x"
            ecov = f"{res['cov_enhanced']:.2f}x"
            print(f"  {name.capitalize():<10} | {loss:<12} | {exp:<10} | {cap:<10} | {icov:<10} | {ecov:<10}")

        # DEFAULT
        print("\nDEFAULT SCENARIOS:")
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

    # Sensitivity
    if verbose:
        print("\n5. PARAMETER SENSITIVITY")
        sens_results = fund.parameter_sensitivity_analysis(n_iterations=2000)
        for p, table in sens_results.items():
            print(f"\nSensitivity: {p.upper()}")
            print(f"{'Value':<10} | {'Alpha (Y5)':<12} | {'Alpha (Y10)':<12}")
            print("-" * 45)
            for row in table:
                print(f"{row['value']:<10.2f} | {row['alpha_y5']:<12.2%} | {row['alpha_y10']:<12.2%}")

    # Plots
    if verbose:
        print("\nGenerating Visualizations...")
        create_visualizations(results, sens_results, fund.simulation_horizon)


if __name__ == "__main__":
    run_dynamic_analysis()