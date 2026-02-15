import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

if __name__ == '__main__':
  # Set random seed for reproducibility
  np.random.seed(42)

  # Generate synthetic regression data
  n_samples = 100
  x = np.linspace(0, 1, n_samples)
  # True underlying function: quadratic
  y_true = 2 * (x - 0.25) ** 2 - 0.1 * x + 1.5
  std_true = np.exp(-2.75 + 2 * x)
  # Add Gaussian noise
  noise = np.random.normal(size=(n_samples, )) * std_true
  y_observed = y_true + noise

  # Define quadratic model: y = a*x^2 + b*x + c
  def quadratic_model(params, x):
    a, b, c = params
    return a * x ** 2 + b * x + c

  # Define loss function (mean squared error)
  def loss_function(params, x, y):
    y_pred = quadratic_model(params, x)
    return np.mean((y - y_pred) ** 2)

  # Initial guess for parameters [a, b, c]
  initial_params = [0.0, 0.0, 0.0]

  # Fit the model using scipy.optimize
  result = minimize(loss_function, initial_params, args=(x, y_observed), method='BFGS')
  fitted_params = result.x
  print(f"Fitted parameters: a={fitted_params[0]:.4f}, b={fitted_params[1]:.4f}, c={fitted_params[2]:.4f}")
  print(f"True parameters: a=0.5, b=-3.0, c=5.0")

  # Generate predictions
  y_pred = quadratic_model(fitted_params, x)

  # ============ GENERAL REGRESSION (with heteroscedastic noise) ============
  # Model both mean and variance as functions of x
  def sigma_model(sigma_params, x):
    """Model sigma as a linear function: sigma(x) = exp(d*x + e)"""
    d, e = sigma_params
    return np.exp(d * x + e)  # exp to ensure positive sigma

  def negative_log_likelihood(params, x, y):
    """Negative log-likelihood for Gaussian with varying sigma"""
    mean_params = params[:3]  # a, b, c
    sigma_params = params[3:]  # d, e

    mu = quadratic_model(mean_params, x)
    sigma = sigma_model(sigma_params, x)

    # NLL for Gaussian: -log p(y|x) = log(sigma) + (y - mu)^2 / (2*sigma^2) + const
    nll = np.sum(np.log(sigma) + (y - mu)**2 / (2 * sigma**2))
    return nll

  # Initial guess: use ordinary regression params + constant sigma
  initial_params_general = np.concatenate([fitted_params, [0.0, np.log(3.0)]])

  # Fit the general model
  result_general = minimize(negative_log_likelihood, initial_params_general,
                           args=(x, y_observed), method='BFGS')
  fitted_params_general = result_general.x
  mean_params_general = fitted_params_general[:3]
  sigma_params_general = fitted_params_general[3:]

  print(f"\nGeneral regression (heteroscedastic):")
  print(f"Mean parameters: a={mean_params_general[0]:.4f}, b={mean_params_general[1]:.4f}, c={mean_params_general[2]:.4f}")
  print(f"Sigma parameters: d={sigma_params_general[0]:.4f}, e={sigma_params_general[1]:.4f}")

  y_pred_general = quadratic_model(mean_params_general, x)
  sigma_pred_general = sigma_model(sigma_params_general, x)

  # ============ CREATE THREE PLOTS ============
  _, axes = plt.subplots(1, 1, figsize=(5, 5))
  axes.scatter(x, y_observed, alpha=0.8, s=30, color='black')
  axes.set_xlabel('Input feature $x$', fontsize=12)
  axes.set_ylabel('Target $y$', fontsize=12)
  axes.set_title('Original Problem', fontsize=14, fontweight='bold')
  axes.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.savefig('imgs/regression-example-1.png', dpi=150, bbox_inches='tight')

  fig, axes = plt.subplots(1, 1, figsize=(5, 5))
  axes.scatter(x, y_observed, alpha=0.8, s=30, label='Observed data', color='black')
  axes.plot(x, y_true, color='blue', linestyle='--', linewidth=2, label='True function')
  axes.plot(x, y_pred, color='red', linewidth=2,
              label=f'Fitted')
  axes.set_xlabel('Input feature $x$', fontsize=12)
  axes.set_ylabel('Target $y$', fontsize=12)
  axes.set_title('Ordinary Regression', fontsize=14, fontweight='bold')
  axes.legend(fontsize=9)
  axes.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.savefig('imgs/regression-example-2.png', dpi=150, bbox_inches='tight')

  fig, axes = plt.subplots(1, 1, figsize=(5, 5))
  axes.scatter(x, y_observed, alpha=0.8, s=30, label='Observed data', color='black')
  axes.plot(x, y_true, color='blue', linestyle='--', linewidth=2, label='True function')
  axes.plot(x, y_pred_general, color='red', linewidth=2,
              label=f'$\\mu(x)$')

  # Plot confidence bands (±2 sigma)
  for i in range(3):
    axes.fill_between(x, y_pred_general - (i + 1)*sigma_pred_general,
                      y_pred_general + (i + 1) *sigma_pred_general,
                      alpha=0.1, color='red', label='$\\pm \\sigma(x)$' if i == 0 else None)

  axes.set_xlabel('Input feature $x$', fontsize=12)
  axes.set_ylabel('Target $y$', fontsize=12)
  axes.set_title('General Regression', fontsize=14, fontweight='bold')
  axes.legend(fontsize=9)
  axes.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.savefig('imgs/regression-example-3.png', dpi=150, bbox_inches='tight')