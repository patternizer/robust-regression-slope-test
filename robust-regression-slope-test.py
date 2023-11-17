#! /usr/bin python

#------------------------------------------------------------------------------
# PROGRAM: robust-regression-slope-test.py
#------------------------------------------------------------------------------
# Version 0.1
# 16 November, 2023
# Michael Taylor
# michael DOT a DOT taylor AT uea DOT ac DOT uk 
#------------------------------------------------------------------------------

# Dataframe libraries:

import numpy as np

# Plotting libraries:
    
import matplotlib.pyplot as plt; plt.close('all')
import seaborn as sns; sns.set()

# Stats libraries:

from scipy import stats
from scipy.stats import t
from scipy.special import erfinv
import statsmodels.api as sm

#------------------------------------------------------------------------------
# SETTINGS
#------------------------------------------------------------------------------

ci = 95

#------------------------------------------------------------------------------
# METHODS
#------------------------------------------------------------------------------

def fit_linear_regression( x, y, method, ci ):
    
    xmean = np.mean(x)
    ymean = np.mean(y) 

    X = np.linspace(np.min(x), np.max(x), n)
    X = sm.add_constant(X)
    if method == 'ols':    
        model = sm.OLS(y, X).fit()
    elif method == 'robust':
        model = sm.RLM(y, X).fit() # Theil-Sen

    y_fit = model.predict(X)    
    
    # COMPUTE: number of standard deviations corresponding to c.i.
        
    alpha = 1 - ( ci / 100 )                    # significance level (0.05)
    percentile = 1 - ( alpha / 2 )              # distribution percentile (0.975)
    n_sd = np.sqrt( 2.0 ) * erfinv( ci / 100 )  # 1.96
    
    # COMPUTE: residual standard error    

    residuals = y - y_fit
    dof = n - 2                                 # degrees of freedom: 2 coeffs --> slope and intercept
    n_sd = stats.t.ppf( percentile, dof )       # 1.98
    sse = np.sum( residuals**2 )                # sum of squared residuals
    se = np.sqrt( sse / dof )                   # residual standard error

    '''
    OLS:
        
    beta_1 = np.cov( x, y )[0, 1] / (np.std( x, ddof = 2)**2)
    beta_0 = ymean - beta_1 * xmean
    y_fit = beta_0 + x * beta_1 
    sse = np.sum( ( y - y_fit ) ** 2)

    t_value = beta_1 / (se / np.sqrt( np.sum( (x - xmean)**2 ) ) )

    p_value_lower = t.cdf( -np.abs( t_value ), dof )
    p_value_upper = 1 - t.cdf( t_value, dof )
    p_value = p_value_lower + p_value_upper
    '''
    
    # COMPUTE: uncertainty on the slope
    
    uncertainty = n_sd * se * np.sqrt( 1/n + (x - xmean)**2 / np.sum( (x - xmean)**2 ) )
    lower_bound = y_fit - uncertainty
    upper_bound = y_fit + uncertainty

    # EXTRACT: model parameters and c.i. on parameters
    
    params = model.params    
    params_ci = model.conf_int(alpha=alpha)    
    pvalues = model.pvalues

    return y_fit, lower_bound, upper_bound, params, params_ci, pvalues

#------------------------------------------------------------------------------
# DATA: create surrogate timeseries
#------------------------------------------------------------------------------

random_state = np.random.seed(42)
n = 50
x = np.arange(n)
#y = 50.0 + 1.5 * x + 1.0 * np.random.normal(0, 10, size=n)
y = 50.0 + 0.4 * x + 1.0 * np.random.normal(0, 10, size=n)

# CREATE: outlier

y[3] = y[3] * 2.0

#------------------------------------------------------------------------------
# FIT: linear regression for both OLS and robust OLS (Theil-Sen)
#------------------------------------------------------------------------------

y_fit_ols, lower_bound_ols, upper_bound_ols, params_ols, params_ci_ols, pvalues_ols = fit_linear_regression( x, y, 'ols', ci )
y_fit_robust, lower_bound_robust, upper_bound_robust, params_robust, params_ci_robust, pvalues_robust = fit_linear_regression( x, y, 'robust', ci )

#------------------------------------------------------------------------------
# Hypothesis test
#------------------------------------------------------------------------------

# 1) Null hypothesis H(0): beta = 0 --> no slope
# 2) Significance level = alpha
# 3) Test statistic: t = beta / sb = beta / ( se / sqrt( sum( (x-xmean)**2 ) ) )
# 4) p-value < alpha? TRUE --> Reject H(0) and accept H(a) that beta is useful predictor of slope
    
alpha = np.round( 1.0 - ( ci / 100.0 ), 3 ) 

hypothesis_test_ols = pvalues_ols[1] < alpha
hypothesis_test_robust = pvalues_robust[1] < alpha

if hypothesis_test_ols == True:
    print('OLS: reject H(0) at alpha=' + str(alpha) + ' : slope is significant (p=' + str( pvalues_ols[1] ) + ')' )
else:
    print('OLS: acceot H(0) at alpha=' + str(alpha) + ' : slope is not significant (p=' + str( pvalues_ols[1] ) + ')' )

if hypothesis_test_robust == True:
    print('Theil-Sen: reject H(0) at alpha=' + str(alpha) + ' : slope is significant (p=' + str( pvalues_robust[1] ) + ')' )
else:
    print('Theil-Sen: acceot H(0) at alpha=' + str(alpha) + ' : slope is not significant (p=' + str( pvalues_robust[1] ) + ')' )

#------------------------------------------------------------------------------
# PLOT
#------------------------------------------------------------------------------

fontsize = 16
dpi = 300

figstr = 'linear-regression-comparison-with-uncertainty.png'
titlestr = 'Linear regressions with uncertainty band'

fig,ax = plt.subplots(figsize=(15,10))
plt.plot(x, y, 'o', color='black', alpha=0.5, label='Data', zorder=1)
plt.plot(x, y_fit_ols, color='blue', lw=1, label='OLS (' + r'$\alpha=$' + str(alpha) + ',' + ' p=' + str( np.round( pvalues_ols[1], 6 ) ) + ')', zorder=3)
plt.plot(x, y_fit_robust, color='red', lw=1, label='Theil-Sen (' + r'$\alpha=$' + str(alpha) + ',' + ' p=' + str( np.round( pvalues_robust[1], 6 ) ) + ')', zorder=3)
plt.fill_between(x, lower_bound_ols, upper_bound_ols, color='blue', alpha=0.1, label='OLS 95% c.i.', zorder=2)
plt.fill_between(x, lower_bound_robust, upper_bound_robust, color='red', alpha=0.1, label='Theil-Sen 95% c.i.', zorder=2)
plt.plot(x[3], y[3], '*', markersize=10, color='black', markerfacecolor=None, label='Outlier')
plt.tick_params(labelsize=fontsize )    
plt.xlabel('x', fontsize=fontsize )
plt.ylabel('y', fontsize=fontsize )
plt.title( titlestr, fontsize=fontsize )
plt.legend( fontsize=fontsize )
plt.savefig( figstr, dpi=dpi )
    
#------------------------------------------------------------------------------
print('** END')


