# Differential Gene Expression Analysis Using Gaussian Process Regression

## Problem Statement

In this project, we aim to determine whether the expression of genes is affected by treatment conditions in time-series data. We'll analyze the GSE102560 dataset, which contains gene expression measurements under different conditions over time. The key challenge is to identify statistically significant differences in gene expression patterns between control and treatment groups.

We'll use Gaussian Process Regression (GPR) to model the time-series data, which will allow us to:
1. Account for the temporal dependencies in gene expression
2. Model uncertainty in our measurements 
3. Create a statistical framework for testing differential expression

## 1. Data Loading and Preprocessing

First, let's import necessary libraries and load the dataset:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

# Set styling for plots
plt.style.use('seaborn-whitegrid')
sns.set_context("notebook", font_scale=1.5)

# Load the dataset
file_path = "GSE102560_brg_all.csv"
df = pd.read_csv(file_path)

# Display basic information
print("Dataset Information:")
print(df.info())
print("\nSample of the Data:")
print(df.head())
print("\nColumn Names:")
print(df.columns)
```

Based on the output shown in the prompt, we can see that the dataset has 35,012 entries and 7 columns. The columns include `rowname` (gene identifiers), `baseMean` (average expression level), `log2FoldChange` (log2 ratio of expression in treatment vs. control), and various statistical measures (`lfcSE`, `stat`, `pvalue`, `padj`).

However, this dataset appears to already contain preprocessed differential expression results rather than raw time-series data. Let's adapt our approach by simulating time-series data based on the given fold-changes, which will allow us to demonstrate the GPR methodology.

## 2. Understanding Gene Expression and Conditions

Gene expression is the process by which information from a gene is used to synthesize gene products, often proteins. The level of gene expression can be measured and represents how active a gene is in a particular cell or tissue.

In gene expression studies:
- **Control condition**: Represents the baseline state (normal conditions)
- **Treatment condition**: Represents the experimental state after applying some stimulus (drug, stress, etc.)

Differential expression analysis aims to identify genes whose expression levels significantly change between these conditions.

## 3. Creating Simulated Time-Series Data

Since the provided dataset contains summarized results rather than raw time series, we'll create simulated time-series data based on the log2FoldChange values:

```python
# Filter data to only include rows with complete information
filtered_df = df.dropna(subset=['log2FoldChange', 'pvalue', 'padj'])

# Select genes for demonstration (mix of significant and non-significant changes)
selected_genes = filtered_df.sort_values('padj').iloc[0:3].append(
    filtered_df.sort_values('padj', ascending=False).iloc[0:3]
)

print("Selected genes for demonstration:")
print(selected_genes[['rowname', 'log2FoldChange', 'pvalue', 'padj']])

# Create time points (e.g., 0, 2, 4, 6, 8, 10, 12 hours)
time_points = np.array([0, 2, 4, 6, 8, 10, 12]).reshape(-1, 1)

# Function to generate simulated time series data for a gene
def generate_time_series(base_expr, log2fc, is_control=True, num_replicates=3):
    # Base expression curve - starts at base level and changes over time
    if is_control:
        # Control condition: relatively stable with slight fluctuations
        mean_expr = base_expr * np.ones_like(time_points)
    else:
        # Treatment condition: gradual change according to log2fc
        factor = 2 ** log2fc  # Convert log2FC to linear fold change
        mean_expr = base_expr * (1 + (factor - 1) * (1 - np.exp(-0.3 * time_points)))
    
    # Add noise and create replicates
    all_series = []
    for i in range(num_replicates):
        # Add biological variation (more at higher expression levels)
        noise = np.random.normal(0, 0.1 * mean_expr, size=mean_expr.shape)
        series = mean_expr + noise
        all_series.append(series)
    
    return np.hstack(all_series)

# Generate data for each selected gene
simulated_data = {}
for _, row in selected_genes.iterrows():
    gene_id = row['rowname']
    base_expr = row['baseMean']
    log2fc = row['log2FoldChange']
    
    # Generate control and treatment data
    control_data = generate_time_series(base_expr, 0, is_control=True)
    treatment_data = generate_time_series(base_expr, log2fc, is_control=False)
    
    simulated_data[gene_id] = {
        'time_points': time_points,
        'control_data': control_data,
        'treatment_data': treatment_data,
        'log2fc': log2fc,
        'pvalue': row['pvalue'],
        'padj': row['padj']
    }

# Plot the simulated data for one gene
def plot_simulated_data(gene_id):
    data = simulated_data[gene_id]
    
    plt.figure(figsize=(12, 6))
    
    # Plot control data
    for i in range(3):  # For each replicate
        plt.scatter(data['time_points'], data['control_data'][:, i], 
                   color='blue', alpha=0.7, label='Control' if i==0 else None)
    
    # Plot treatment data
    for i in range(3):  # For each replicate
        plt.scatter(data['time_points'], data['treatment_data'][:, i], 
                   color='red', alpha=0.7, label='Treatment' if i==0 else None)
    
    plt.title(f"Simulated Expression Data for {gene_id}\nlog2FC: {data['log2fc']:.3f}, p-adj: {data['padj']:.3e}")
    plt.xlabel("Time (hours)")
    plt.ylabel("Expression Level")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot simulated data for the first significant gene
plot_simulated_data(selected_genes['rowname'].iloc[0])
```

## 4. Gaussian Process Regression (GPR): Mathematical Foundation

Gaussian Process Regression is a non-parametric Bayesian approach to regression that defines a prior distribution over functions and then updates this prior with observed data to get a posterior distribution over functions.

### Mathematical Foundation

A Gaussian Process (GP) is defined as a collection of random variables, any finite number of which have a joint Gaussian distribution. It is completely specified by:

1. A mean function $m(x)$
2. A covariance function (kernel) $k(x, x')$

For a function $f(x)$ modeled as a GP:

$$f(x) \sim \mathcal{GP}(m(x), k(x, x'))$$

Where:
- $m(x) = \mathbb{E}[f(x)]$ is the mean function
- $k(x, x') = \mathbb{E}[(f(x) - m(x))(f(x') - m(x'))]$ is the covariance function

### Posterior Prediction

Given training data $\mathbf{X}$ with outputs $\mathbf{y}$, and test points $\mathbf{X_*}$, the predictive distribution for $\mathbf{f_*}$ is:

$$p(\mathbf{f_*}|\mathbf{X_*}, \mathbf{X}, \mathbf{y}) = \mathcal{N}(\mathbf{\mu_*}, \mathbf{\Sigma_*})$$

Where:
- $\mathbf{\mu_*} = m(\mathbf{X_*}) + K(\mathbf{X_*}, \mathbf{X})[K(\mathbf{X}, \mathbf{X}) + \sigma_n^2 I]^{-1}(\mathbf{y} - m(\mathbf{X}))$
- $\mathbf{\Sigma_*} = K(\mathbf{X_*}, \mathbf{X_*}) - K(\mathbf{X_*}, \mathbf{X})[K(\mathbf{X}, \mathbf{X}) + \sigma_n^2 I]^{-1}K(\mathbf{X}, \mathbf{X_*})$

Where $K$ is the kernel matrix and $\sigma_n^2$ represents the variance of the observation noise.

### Relevance to Gene Expression Time Series

GPR is particularly suitable for time-series gene expression data because:

1. It naturally handles uneven sampling and missing data points
2. It provides uncertainty quantification (confidence intervals)
3. It can model complex temporal patterns with appropriate kernel functions
4. It allows for incorporation of prior biological knowledge through kernel design

## 5. Implementing GPR for Gene Expression Analysis

Let's implement GPR for our simulated gene expression data:

```python
def fit_gp(X, y, kernel=None):
    """Fit a Gaussian Process Regressor to data"""
    if kernel is None:
        # Define a suitable kernel for time-series gene expression
        # Combination of RBF (smoothness), WhiteKernel (noise), and ConstantKernel (scale)
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
    
    # Create and fit the model
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-10)
    gpr.fit(X, y)
    
    return gpr

def predict_gp(gpr, X_new):
    """Make predictions with a fitted GP model"""
    y_pred, y_std = gpr.predict(X_new, return_std=True)
    return y_pred, y_std

def plot_gp_fit(gene_id, condition, X, y, gpr):
    """Plot GP fit for a specific condition"""
    # Create dense time points for smooth prediction
    X_dense = np.linspace(0, 12, 100).reshape(-1, 1)
    
    # Predict with the GP model
    y_pred, y_std = predict_gp(gpr, X_dense)
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    # Plot data points
    colors = {'control': 'blue', 'treatment': 'red'}
    color = colors[condition]
    
    # For each replicate
    for i in range(y.shape[1]):
        plt.scatter(X, y[:, i], color=color, alpha=0.7, 
                   label=f'{condition.capitalize()} Data' if i==0 else None)
    
    # Plot GP prediction
    plt.plot(X_dense, y_pred, color=color, linestyle='-', label=f'{condition.capitalize()} GP Fit')
    
    # Plot confidence intervals (2 standard deviations)
    plt.fill_between(X_dense.ravel(), y_pred - 2*y_std, y_pred + 2*y_std,
                    alpha=0.2, color=color, label='95% Confidence Interval')
    
    plt.title(f"GP Regression for {gene_id} - {condition.capitalize()} Condition")
    plt.xlabel("Time (hours)")
    plt.ylabel("Expression Level")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return gpr, X_dense, y_pred, y_std

# Fit GP models to each gene's data
gp_models = {}

for gene_id, data in simulated_data.items():
    # Get data
    X = data['time_points']
    y_control = data['control_data']
    y_treatment = data['treatment_data']
    
    # Fit GP for control
    control_gpr = fit_gp(X, y_control)
    
    # Fit GP for treatment
    treatment_gpr = fit_gp(X, y_treatment)
    
    # Store models
    gp_models[gene_id] = {
        'control_gpr': control_gpr,
        'treatment_gpr': treatment_gpr
    }

# Plot GP fits for the first gene
gene_id = selected_genes['rowname'].iloc[0]
data = simulated_data[gene_id]
X = data['time_points']
y_control = data['control_data']
y_treatment = data['treatment_data']

# Plot control fit
control_gpr, X_dense, y_control_pred, y_control_std = plot_gp_fit(
    gene_id, 'control', X, y_control, gp_models[gene_id]['control_gpr'])

# Plot treatment fit
treatment_gpr, X_dense, y_treatment_pred, y_treatment_std = plot_gp_fit(
    gene_id, 'treatment', X, y_treatment, gp_models[gene_id]['treatment_gpr'])
```

## 6. Kernel Functions and Their Role in GPR

The kernel function in GPR determines the shape and properties of the functions that can be modeled. Different kernels capture different assumptions about the underlying process:

```python
def plot_kernel_functions():
    """Plot different kernel functions and their impact on GP regression"""
    # Create sample data
    X = np.linspace(0, 10, 10).reshape(-1, 1)
    y = np.sin(X).ravel() + 0.1 * np.random.randn(10)
    
    # Define different kernels
    kernels = {
        'RBF (Squared Exponential)': RBF(length_scale=1.0),
        'RBF + White Noise': RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1),
        'Scaled RBF': ConstantKernel(1.0) * RBF(length_scale=1.0),
        'Combined (RBF + White Noise + Constant)': ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    }
    
    # Create test points for prediction
    X_test = np.linspace(0, 10, 100).reshape(-1, 1)
    
    # Plot each kernel
    plt.figure(figsize=(15, 12))
    
    for i, (name, kernel) in enumerate(kernels.items()):
        plt.subplot(2, 2, i+1)
        
        # Fit GP with this kernel
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        gpr.fit(X, y)
        
        # Make predictions
        y_pred, y_std = gpr.predict(X_test, return_std=True)
        
        # Plot data and predictions
        plt.scatter(X, y, color='black', label='Observations')
        plt.plot(X_test, y_pred, color='blue', label='Prediction')
        plt.fill_between(X_test.ravel(), y_pred - 2*y_std, y_pred + 2*y_std,
                        alpha=0.2, color='blue', label='95% Confidence Interval')
        
        plt.title(f"Kernel: {name}\nLog-Likelihood: {gpr.log_marginal_likelihood_value_:.2f}")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.legend()
    
    plt.tight_layout()
    plt.show()

# Plot different kernel functions
plot_kernel_functions()
```

### Kernel Selection for Gene Expression Time Series

For gene expression time series, we typically use a combination of kernels:

1. **RBF (Squared Exponential) kernel**: Models smooth trends in expression
2. **WhiteKernel**: Accounts for noise in measurements
3. **ConstantKernel**: Scales the overall magnitude of variation

The optimal kernel parameters are learned during model fitting by maximizing the marginal likelihood.

## 7. Making Predictions with GPR Models

We can use the fitted GPR models to make predictions about expression levels at any time point, including those not in our original data:

```python
def compare_predictions(gene_id, data, gp_models):
    """Compare predictions from control and treatment models"""
    # Get models
    control_gpr = gp_models[gene_id]['control_gpr']
    treatment_gpr = gp_models[gene_id]['treatment_gpr']
    
    # Create dense time points
    X_dense = np.linspace(0, 12, 100).reshape(-1, 1)
    
    # Predict with both models
    y_control_pred, y_control_std = predict_gp(control_gpr, X_dense)
    y_treatment_pred, y_treatment_std = predict_gp(treatment_gpr, X_dense)
    
    # Calculate log2 fold change at each time point
    log2fc_pred = np.log2(y_treatment_pred / y_control_pred)
    
    # Create dataframe for easy plotting
    pred_df = pd.DataFrame({
        'Time': X_dense.ravel(),
        'Control': y_control_pred,
        'Control_lower': y_control_pred - 2*y_control_std,
        'Control_upper': y_control_pred + 2*y_control_std,
        'Treatment': y_treatment_pred,
        'Treatment_lower': y_treatment_pred - 2*y_treatment_std,
        'Treatment_upper': y_treatment_pred + 2*y_treatment_std,
        'log2FC': log2fc_pred
    })
    
    # Plot expression levels
    plt.figure(figsize=(15, 10))
    
    # Plot expression profiles
    plt.subplot(2, 1, 1)
    plt.plot(pred_df['Time'], pred_df['Control'], color='blue', label='Control')
    plt.fill_between(pred_df['Time'], pred_df['Control_lower'], pred_df['Control_upper'],
                    alpha=0.2, color='blue')
    
    plt.plot(pred_df['Time'], pred_df['Treatment'], color='red', label='Treatment')
    plt.fill_between(pred_df['Time'], pred_df['Treatment_lower'], pred_df['Treatment_upper'],
                    alpha=0.2, color='red')
    
    plt.title(f"Predicted Expression Profiles for {gene_id}\nTrue log2FC: {data['log2fc']:.3f}, p-adj: {data['padj']:.3e}")
    plt.xlabel("Time (hours)")
    plt.ylabel("Expression Level")
    plt.legend()
    plt.grid(True)
    
    # Plot log2 fold change
    plt.subplot(2, 1, 2)
    plt.plot(pred_df['Time'], pred_df['log2FC'], color='purple')
    plt.axhline(y=0, color='black', linestyle='--')
    
    plt.title(f"Predicted log2 Fold Change over Time")
    plt.xlabel("Time (hours)")
    plt.ylabel("log2 Fold Change")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return pred_df

# Compare predictions for the first gene
gene_id = selected_genes['rowname'].iloc[0]
pred_df = compare_predictions(gene_id, simulated_data[gene_id], gp_models)
```

### Time-Dependent Differential Expression

One of the advantages of using GPR for time-series gene expression data is the ability to examine how differential expression changes over time:

```python
def plot_time_dependent_de(gene_ids, gp_models, simulated_data):
    """Plot time-dependent differential expression for multiple genes"""
    # Create dense time points
    X_dense = np.linspace(0, 12, 100).reshape(-1, 1)
    
    plt.figure(figsize=(12, 8))
    
    for gene_id in gene_ids:
        # Get models
        control_gpr = gp_models[gene_id]['control_gpr']
        treatment_gpr = gp_models[gene_id]['treatment_gpr']
        
        # Predict with both models
        y_control_pred, _ = predict_gp(control_gpr, X_dense)
        y_treatment_pred, _ = predict_gp(treatment_gpr, X_dense)
        
        # Calculate log2 fold change
        log2fc_pred = np.log2(y_treatment_pred / y_control_pred)
        
        # Plot
        plt.plot(X_dense, log2fc_pred, label=f"{gene_id} (adj-p: {simulated_data[gene_id]['padj']:.2e})")
    
    plt.axhline(y=0, color='black', linestyle='--')
    plt.title("Time-Dependent Differential Expression")
    plt.xlabel("Time (hours)")
    plt.ylabel("log2 Fold Change")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot time-dependent DE for the first few genes
gene_ids = selected_genes['rowname'].values[:3]
plot_time_dependent_de(gene_ids, gp_models, simulated_data)
```

## 8. Differential Expression Analysis Using GPR

Now, let's perform a formal differential expression analysis using our GPR models:

```python
def detect_differential_expression(gene_id, gp_models, alpha=0.05, num_points=100):
    """
    Detect differential expression using GPR models
    
    Returns:
    - is_de: Boolean indicating if gene is differentially expressed
    - max_diff: Maximum absolute difference
    - max_diff_time: Time point of maximum difference
    - pvalue: p-value for the test
    """
    # Get models
    control_gpr = gp_models[gene_id]['control_gpr']
    treatment_gpr = gp_models[gene_id]['treatment_gpr']
    
    # Create dense time points
    X_dense = np.linspace(0, 12, num_points).reshape(-1, 1)
    
    # Predict with both models
    y_control_pred, y_control_std = predict_gp(control_gpr, X_dense)
    y_treatment_pred, y_treatment_std = predict_gp(treatment_gpr, X_dense)
    
    # Calculate differences and standard errors
    diff = y_treatment_pred - y_control_pred
    # Combined variance (assuming independence)
    var_combined = y_control_std**2 + y_treatment_std**2
    se_combined = np.sqrt(var_combined)
    
    # Find point of maximum absolute difference
    max_diff_idx = np.argmax(np.abs(diff))
    max_diff = diff[max_diff_idx]
    max_diff_time = X_dense[max_diff_idx][0]
    
    # Perform t-test at each time point
    t_stats = diff / se_combined
    p_values = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))
    
    # Correct for multiple testing (Benjamini-Hochberg)
    reject, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    
    # Gene is DE if any time point is significant after correction
    is_de = np.any(reject)
    min_pvalue = np.min(p_corrected)
    
    return {
        'is_de': is_de,
        'max_diff': max_diff,
        'max_diff_time': max_diff_time,
        't_stats': t_stats,
        'p_values': p_values,
        'p_corrected': p_corrected,
        'min_pvalue': min_pvalue
    }

def plot_differential_expression_test(gene_id, de_results, gp_models):
    """Plot differential expression test results"""
    # Get models
    control_gpr = gp_models[gene_id]['control_gpr']
    treatment_gpr = gp_models[gene_id]['treatment_gpr']
    
    # Create dense time points
    X_dense = np.linspace(0, 12, 100).reshape(-1, 1)
    
    # Predict with both models
    y_control_pred, y_control_std = predict_gp(control_gpr, X_dense)
    y_treatment_pred, y_treatment_std = predict_gp(treatment_gpr, X_dense)
    
    # Calculate differences
    diff = y_treatment_pred - y_control_pred
    
    plt.figure(figsize=(15, 12))
    
    # Plot expression profiles
    plt.subplot(3, 1, 1)
    
    plt.plot(X_dense, y_control_pred, color='blue', label='Control')
    plt.fill_between(X_dense.ravel(), 
                    y_control_pred - 2*y_control_std, 
                    y_control_pred + 2*y_control_std,
                    alpha=0.2, color='blue')
    
    plt.plot(X_dense, y_treatment_pred, color='red', label='Treatment')
    plt.fill_between(X_dense.ravel(), 
                    y_treatment_pred - 2*y_treatment_std, 
                    y_treatment_pred + 2*y_treatment_std,
                    alpha=0.2, color='red')
    
    plt.title(f"Gene: {gene_id} - {'Differentially Expressed' if de_results['is_de'] else 'Not Differentially Expressed'}")
    plt.xlabel("Time (hours)")
    plt.ylabel("Expression Level")
    plt.legend()
    plt.grid(True)
    
    # Plot difference
    plt.subplot(3, 1, 2)
    plt.plot(X_dense, diff, color='purple')
    plt.axhline(y=0, color='black', linestyle='--')
    
    # Mark the point of maximum difference
    plt.scatter(de_results['max_diff_time'], de_results['max_diff'], 
               color='red', s=100, zorder=5, 
               label=f"Max diff: {de_results['max_diff']:.2f} at t={de_results['max_diff_time']:.1f}")
    
    plt.title("Difference (Treatment - Control)")
    plt.xlabel("Time (hours)")
    plt.ylabel("Difference")
    plt.legend()
    plt.grid(True)
    
    # Plot p-values
    plt.subplot(3, 1, 3)
    plt.semilogy(X_dense, de_results['p_corrected'], color='green')
    plt.axhline(y=0.05, color='red', linestyle='--', label='α = 0.05')
    
    plt.title(f"FDR-Corrected p-values (min: {de_results['min_pvalue']:.2e})")
    plt.xlabel("Time (hours)")
    plt.ylabel("p-value (log scale)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Perform differential expression analysis for all genes
de_results = {}

for gene_id in simulated_data.keys():
    de_results[gene_id] = detect_differential_expression(gene_id, gp_models)

# Plot results for the first gene
gene_id = selected_genes['rowname'].iloc[0]
plot_differential_expression_test(gene_id, de_results[gene_id], gp_models)
```

## 9. Analyzing the Covariance Function (Gaussian Process Prior)

The covariance function (kernel) in a GP defines our prior beliefs about the function we're modeling. Let's examine the learned kernels for our gene expression data:

```python
def analyze_covariance_function(gene_id, gp_models):
    """Analyze the covariance function (kernel) for a gene"""
    control_gpr = gp_models[gene_id]['control_gpr']
    treatment_gpr = gp_models[gene_id]['treatment_gpr']
    
    print(f"Gene: {gene_id}")
    print("\nControl Condition Kernel:")
    print(control_gpr.kernel_)
    print(f"Log-Likelihood: {control_gpr.log_marginal_likelihood_value_:.4f}")
    
    print("\nTreatment Condition Kernel:")
    print(treatment_gpr.kernel_)
    print(f"Log-Likelihood: {treatment_gpr.log_marginal_likelihood_value_:.4f}")
    
    # Visualize the kernel function
    plt.figure(figsize=(12, 6))
    
    # Define distance points
    dist = np.linspace(0, 10, 100)
    
    # Compute kernel values
    control_k = np.zeros_like(dist)
    treatment_k = np.zeros_like(dist)
    
    for i, d in enumerate(dist):
        x1 = np.array([[0]])
        x2 = np.array([[d]])
        control_k[i] = control_gpr.kernel_(x1, x2)[0][0]
        treatment_k[i] = treatment_gpr.kernel_(x1, x2)[0][0]
    
    # Plot
    plt.plot(dist, control_k, label='Control Kernel', color='blue')
    plt.plot(dist, treatment_k, label='Treatment Kernel', color='red')
    
    plt.title(f"Kernel Functions for Gene {gene_id}")
    plt.xlabel("Distance between time points")
    plt.ylabel("Covariance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Analyze kernels for the first gene
gene_id = selected_genes['rowname'].iloc[0]
analyze_covariance_function(gene_id, gp_models)
```

## 10. Summary of Differential Expression Results

Finally, let's summarize our findings across all genes:

```python
def summarize_results(de_results, simulated_data):
    """Summarize differential expression results"""
    # Create results dataframe
    results = []
    
    for gene_id, result in de_results.items():
        results.append({
            'gene_id': gene_id,
            'is_de': result['is_de'],
            'max_diff': result['max_diff'],
            'max_diff_time': result['max_diff_time'],
            'min_pvalue': result['min_pvalue'],
            'true_log2fc': simulated_data[gene_id]['log2fc'],
            'true_padj': simulated_data[gene_id]['padj']
        })
    
    results_df = pd.DataFrame(results)
    
    # Sort by p-value
    results_df = results_df.sort_values('min_