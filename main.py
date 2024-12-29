import numpy as np
import cvxpy as cp

class LogOptimalPortfolio:
    def __init__(self, returns, transaction_costs, leverage):
        self.returns = returns  # Expected returns matrix (m x n)
        self.transaction_costs = transaction_costs  
        self.leverage = leverage  
        self.n_assets = returns.shape[1]

    def optimize_portfolio(self):
        weights = cp.Variable(self.n_assets)

        # Objective: maximize expected log growth rate
        expected_log_growth = cp.sum(cp.log(1 + self.returns @ weights))

        # Constraints
        constraints = [
            cp.sum(weights) <= self.leverage,  # Leverage constraint
            weights >= 0  # No short selling
        ]
        # Solve
        problem = cp.Problem(cp.Maximize(expected_log_growth), constraints)
        problem.solve(solver=cp.ECOS)

        if problem.status == cp.OPTIMAL:
            optimal_weights = weights.value
            return optimal_weights
        else:
            raise ValueError("Optimization failed.")

# Example usage
if __name__ == "__main__":
    # Simulated returns for 3 assets over 5 scenarios
    simulated_returns = np.array([[0.1, 0.2, 0.15],
                                   [0.05, 0.1, 0.2],
                                   [0.2, 0.15, 0.1],
                                   [0.1, 0.3, 0.25],
                                   [0.15, 0.1, 0.2]])

    transaction_costs = np.array([0.01, 0.01, 0.01])  # Example transaction costs
    leverage_factor = 1.5

    portfolio_optimizer = LogOptimalPortfolio(simulated_returns, transaction_costs, leverage_factor)
    
    optimal_weights = portfolio_optimizer.optimize_portfolio()
    print("Optimal Portfolio Weights:", optimal_weights)
