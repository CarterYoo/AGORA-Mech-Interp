"""
Statistical tests module.

This module provides statistical hypothesis testing and confidence interval
computation for rigorous evaluation of hallucination detection methods.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from loguru import logger
from scipy import stats
from scipy.stats import (
    ttest_ind,
    ttest_rel,
    mannwhitneyu,
    wilcoxon,
    chi2_contingency,
    fisher_exact
)


class StatisticalTests:
    """
    Statistical hypothesis testing for hallucination detection evaluation.
    
    Performs t-tests, non-parametric tests, effect size calculations,
    and confidence interval estimation.
    """
    
    def __init__(
        self,
        alpha: float = 0.05,
        bonferroni_correction: bool = True
    ):
        """
        Initialize statistical tests.
        
        Args:
            alpha: Significance level
            bonferroni_correction: Whether to apply Bonferroni correction
        """
        self.alpha = alpha
        self.bonferroni_correction = bonferroni_correction
        
        logger.info(
            f"Initialized StatisticalTests: alpha={alpha}, "
            f"bonferroni={bonferroni_correction}"
        )
    
    def independent_ttest(
        self,
        group1: List[float],
        group2: List[float],
        equal_var: bool = False
    ) -> Dict:
        """
        Perform independent samples t-test (Welch's t-test by default).
        
        Args:
            group1: First group of values
            group2: Second group of values
            equal_var: Whether to assume equal variance
        
        Returns:
            Dictionary with test results
        """
        if len(group1) < 2 or len(group2) < 2:
            logger.error("Insufficient samples for t-test")
            return {}
        
        t_stat, p_value = ttest_ind(
            group1, group2,
            equal_var=equal_var
        )
        
        mean1 = np.mean(group1)
        mean2 = np.mean(group2)
        mean_diff = mean1 - mean2
        
        cohens_d = self.compute_cohens_d(group1, group2)
        
        result = {
            'test': 'independent_ttest' if equal_var else 'welchs_ttest',
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'mean_group1': float(mean1),
            'mean_group2': float(mean2),
            'mean_difference': float(mean_diff),
            'cohens_d': cohens_d,
            'significant': p_value < self.alpha,
            'n_group1': len(group1),
            'n_group2': len(group2)
        }
        
        logger.info(
            f"T-test: t={t_stat:.3f}, p={p_value:.4f}, "
            f"d={cohens_d:.3f}"
        )
        
        return result
    
    def paired_ttest(
        self,
        before: List[float],
        after: List[float]
    ) -> Dict:
        """
        Perform paired samples t-test.
        
        Args:
            before: Values before treatment
            after: Values after treatment
        
        Returns:
            Dictionary with test results
        """
        if len(before) != len(after):
            logger.error("Paired samples must have equal length")
            return {}
        
        if len(before) < 2:
            logger.error("Insufficient samples for paired t-test")
            return {}
        
        t_stat, p_value = ttest_rel(before, after)
        
        mean_before = np.mean(before)
        mean_after = np.mean(after)
        mean_diff = mean_before - mean_after
        
        differences = np.array(before) - np.array(after)
        cohens_d = float(np.mean(differences) / np.std(differences, ddof=1))
        
        result = {
            'test': 'paired_ttest',
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'mean_before': float(mean_before),
            'mean_after': float(mean_after),
            'mean_difference': float(mean_diff),
            'cohens_d': cohens_d,
            'significant': p_value < self.alpha,
            'n_pairs': len(before)
        }
        
        logger.info(
            f"Paired t-test: t={t_stat:.3f}, p={p_value:.4f}"
        )
        
        return result
    
    def mann_whitney_u(
        self,
        group1: List[float],
        group2: List[float],
        alternative: str = 'two-sided'
    ) -> Dict:
        """
        Perform Mann-Whitney U test (non-parametric).
        
        Args:
            group1: First group of values
            group2: Second group of values
            alternative: Alternative hypothesis ('two-sided', 'less', 'greater')
        
        Returns:
            Dictionary with test results
        """
        if len(group1) < 2 or len(group2) < 2:
            logger.error("Insufficient samples for Mann-Whitney U test")
            return {}
        
        u_stat, p_value = mannwhitneyu(
            group1, group2,
            alternative=alternative
        )
        
        result = {
            'test': 'mann_whitney_u',
            'u_statistic': float(u_stat),
            'p_value': float(p_value),
            'median_group1': float(np.median(group1)),
            'median_group2': float(np.median(group2)),
            'significant': p_value < self.alpha,
            'n_group1': len(group1),
            'n_group2': len(group2)
        }
        
        logger.info(
            f"Mann-Whitney U: U={u_stat:.3f}, p={p_value:.4f}"
        )
        
        return result
    
    def wilcoxon_signed_rank(
        self,
        before: List[float],
        after: List[float]
    ) -> Dict:
        """
        Perform Wilcoxon signed-rank test (non-parametric paired test).
        
        Args:
            before: Values before treatment
            after: Values after treatment
        
        Returns:
            Dictionary with test results
        """
        if len(before) != len(after):
            logger.error("Paired samples must have equal length")
            return {}
        
        w_stat, p_value = wilcoxon(before, after)
        
        result = {
            'test': 'wilcoxon_signed_rank',
            'w_statistic': float(w_stat),
            'p_value': float(p_value),
            'median_before': float(np.median(before)),
            'median_after': float(np.median(after)),
            'significant': p_value < self.alpha,
            'n_pairs': len(before)
        }
        
        logger.info(
            f"Wilcoxon: W={w_stat:.3f}, p={p_value:.4f}"
        )
        
        return result
    
    def chi_square_test(
        self,
        contingency_table: np.ndarray
    ) -> Dict:
        """
        Perform chi-square test of independence.
        
        Args:
            contingency_table: 2D array of observed frequencies
        
        Returns:
            Dictionary with test results
        """
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        result = {
            'test': 'chi_square',
            'chi2_statistic': float(chi2),
            'p_value': float(p_value),
            'degrees_of_freedom': int(dof),
            'significant': p_value < self.alpha
        }
        
        logger.info(
            f"Chi-square: χ²={chi2:.3f}, p={p_value:.4f}, df={dof}"
        )
        
        return result
    
    def fishers_exact_test(
        self,
        table_2x2: np.ndarray
    ) -> Dict:
        """
        Perform Fisher's exact test for 2x2 contingency tables.
        
        Args:
            table_2x2: 2x2 contingency table
        
        Returns:
            Dictionary with test results
        """
        if table_2x2.shape != (2, 2):
            logger.error("Fisher's exact test requires 2x2 table")
            return {}
        
        odds_ratio, p_value = fisher_exact(table_2x2)
        
        result = {
            'test': 'fishers_exact',
            'odds_ratio': float(odds_ratio),
            'p_value': float(p_value),
            'significant': p_value < self.alpha
        }
        
        logger.info(
            f"Fisher's exact: OR={odds_ratio:.3f}, p={p_value:.4f}"
        )
        
        return result
    
    def compute_cohens_d(
        self,
        group1: List[float],
        group2: List[float]
    ) -> float:
        """
        Compute Cohen's d effect size.
        
        Args:
            group1: First group of values
            group2: Second group of values
        
        Returns:
            Cohen's d value
        """
        mean1 = np.mean(group1)
        mean2 = np.mean(group2)
        
        var1 = np.var(group1, ddof=1)
        var2 = np.var(group2, ddof=1)
        n1 = len(group1)
        n2 = len(group2)
        
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        cohens_d = (mean1 - mean2) / pooled_std
        
        return float(cohens_d)
    
    def bootstrap_confidence_interval(
        self,
        data: List[float],
        statistic: str = 'mean',
        n_bootstrap: int = 10000,
        confidence_level: float = 0.95
    ) -> Dict:
        """
        Compute bootstrap confidence interval.
        
        Args:
            data: Data to bootstrap
            statistic: Statistic to compute ('mean', 'median', 'std')
            n_bootstrap: Number of bootstrap iterations
            confidence_level: Confidence level (e.g., 0.95 for 95%)
        
        Returns:
            Dictionary with CI results
        """
        if len(data) < 2:
            logger.error("Insufficient data for bootstrap")
            return {}
        
        if statistic == 'mean':
            stat_func = np.mean
        elif statistic == 'median':
            stat_func = np.median
        elif statistic == 'std':
            stat_func = np.std
        else:
            logger.error(f"Unknown statistic: {statistic}")
            return {}
        
        bootstrap_samples = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_samples.append(stat_func(sample))
        
        lower_percentile = (1 - confidence_level) / 2 * 100
        upper_percentile = (1 + confidence_level) / 2 * 100
        
        ci_lower = np.percentile(bootstrap_samples, lower_percentile)
        ci_upper = np.percentile(bootstrap_samples, upper_percentile)
        
        point_estimate = stat_func(data)
        
        result = {
            'statistic': statistic,
            'point_estimate': float(point_estimate),
            'confidence_level': confidence_level,
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n_bootstrap': n_bootstrap,
            'n_data': len(data)
        }
        
        logger.info(
            f"Bootstrap CI ({confidence_level*100:.0f}%): "
            f"{point_estimate:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]"
        )
        
        return result
    
    def multiple_comparison_correction(
        self,
        p_values: List[float],
        method: str = 'bonferroni'
    ) -> Dict:
        """
        Apply multiple comparison correction.
        
        Args:
            p_values: List of p-values
            method: Correction method ('bonferroni' or 'fdr_bh')
        
        Returns:
            Dictionary with corrected p-values
        """
        n_tests = len(p_values)
        
        if method == 'bonferroni':
            corrected_alpha = self.alpha / n_tests
            corrected_p_values = [min(p * n_tests, 1.0) for p in p_values]
        
        elif method == 'fdr_bh':
            from statsmodels.stats.multitest import multipletests
            _, corrected_p_values, _, _ = multipletests(
                p_values, alpha=self.alpha, method='fdr_bh'
            )
            corrected_p_values = corrected_p_values.tolist()
            corrected_alpha = self.alpha
        
        else:
            logger.error(f"Unknown correction method: {method}")
            return {}
        
        significant = [p < corrected_alpha for p in corrected_p_values]
        
        result = {
            'method': method,
            'n_tests': n_tests,
            'original_alpha': self.alpha,
            'corrected_alpha': corrected_alpha,
            'original_p_values': p_values,
            'corrected_p_values': corrected_p_values,
            'significant': significant,
            'n_significant': sum(significant)
        }
        
        logger.info(
            f"{method.upper()} correction: "
            f"{sum(significant)}/{n_tests} significant after correction"
        )
        
        return result
    
    def effect_size_interpretation(
        self,
        cohens_d: float
    ) -> str:
        """
        Interpret Cohen's d effect size.
        
        Args:
            cohens_d: Cohen's d value
        
        Returns:
            Interpretation string
        """
        abs_d = abs(cohens_d)
        
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def compare_groups_comprehensive(
        self,
        group1: List[float],
        group2: List[float],
        group1_name: str = "Group 1",
        group2_name: str = "Group 2",
        parametric: bool = True
    ) -> Dict:
        """
        Comprehensive comparison of two groups.
        
        Args:
            group1: First group
            group2: Second group
            group1_name: Name of first group
            group2_name: Name of second group
            parametric: Whether to use parametric tests
        
        Returns:
            Comprehensive comparison dictionary
        """
        comparison = {
            'groups': {
                group1_name: {
                    'n': len(group1),
                    'mean': float(np.mean(group1)),
                    'std': float(np.std(group1, ddof=1)),
                    'median': float(np.median(group1))
                },
                group2_name: {
                    'n': len(group2),
                    'mean': float(np.mean(group2)),
                    'std': float(np.std(group2, ddof=1)),
                    'median': float(np.median(group2))
                }
            }
        }
        
        if parametric:
            test_result = self.independent_ttest(group1, group2, equal_var=False)
        else:
            test_result = self.mann_whitney_u(group1, group2)
        
        comparison['statistical_test'] = test_result
        
        group1_ci = self.bootstrap_confidence_interval(group1)
        group2_ci = self.bootstrap_confidence_interval(group2)
        
        comparison['confidence_intervals'] = {
            group1_name: group1_ci,
            group2_name: group2_ci
        }
        
        if parametric:
            effect_size = test_result['cohens_d']
            comparison['effect_size'] = {
                'cohens_d': effect_size,
                'interpretation': self.effect_size_interpretation(effect_size)
            }
        
        logger.info(
            f"Comprehensive comparison: {group1_name} vs {group2_name} complete"
        )
        
        return comparison


def main():
    """
    Example usage of StatisticalTests.
    """
    stat_tests = StatisticalTests(alpha=0.05)
    
    group1 = [0.45, 0.52, 0.38, 0.41, 0.49, 0.44, 0.51]
    group2 = [0.28, 0.35, 0.22, 0.31, 0.29, 0.24, 0.33]
    
    print("Independent samples t-test...")
    ttest_result = stat_tests.independent_ttest(group1, group2)
    print(f"t-statistic: {ttest_result['t_statistic']:.3f}")
    print(f"p-value: {ttest_result['p_value']:.4f}")
    print(f"Cohen's d: {ttest_result['cohens_d']:.3f}")
    print(f"Significant: {ttest_result['significant']}")
    
    print("\nBootstrap confidence interval...")
    ci_result = stat_tests.bootstrap_confidence_interval(group1)
    print(f"Mean: {ci_result['point_estimate']:.3f}")
    print(f"95% CI: [{ci_result['ci_lower']:.3f}, {ci_result['ci_upper']:.3f}]")
    
    print("\nComprehensive comparison...")
    comparison = stat_tests.compare_groups_comprehensive(
        group1, group2,
        group1_name="No Hallucination",
        group2_name="Hallucination"
    )
    print(f"Mean difference: {comparison['groups']['No Hallucination']['mean'] - comparison['groups']['Hallucination']['mean']:.3f}")
    print(f"Effect size: {comparison['effect_size']['interpretation']}")


if __name__ == "__main__":
    main()

