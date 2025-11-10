import numpy as np
from scipy.stats import norm
from scipy.stats import t


np.random.seed(0)           #for reproducible results
N = 10000                   #number of samples
nu = 4                      #degrees of freedom student-t distribution
mean = np.zeros(3)          #mean of distribution dS_2, dS_10, dg
std_dev_2 = 0.0006          #standard deviation dS_2
std_dev_10 = 0.0004         #standard deviation dS_10
std_dev_g = 0.00015         #standard deviation dg
w_2 = 0.7                   #weight 2 year bond
w_10 = -1                   #weight 10 year swap
w_g = 0.3                   #weight green bond
alpha = [0.99,0.975,0.95]   #alphas for VaR and ES and worst quantile

Rho_s_matrix = np.array([[1,0.75,0.25],[0.75,1,0.35],[0.25,0.35,1]]) #Rank correlations matrix dS_2, dS_10,dg
Rho_matrix = 2*np.sin(Rho_s_matrix*np.pi/6) #transform rank correlation to Pearson correlation matrix
print("correlation_matrix",Rho_matrix)

#correlated normal random variables, variance = 1
dS_2, dS_10, dg = np.random.multivariate_normal(mean,Rho_matrix,size = N).T

#check if correlation is right (check with a lot of samples 10^7 for example)
print("correlation",np.corrcoef(dS_2,dS_10)[0,1])
print("correlation",np.corrcoef(dS_10,dg)[0,1])
print("correlation",np.corrcoef(dS_2,dg)[0,1])

#Gaussian copula Gaussian marginal distribution
G_dS_2 = dS_2 *std_dev_2
G_dS_10 = dS_10*std_dev_10
G_dg = dg*std_dev_g

#print the results
print("G_2",G_dS_2)
print("G_10",G_dS_10)
print("G_g",G_dg)

#student-t copula with 4 degrees of freedom
chi_squared = np.random.chisquare(df=nu,size=N)
chi_adjusted = np.sqrt(chi_squared/nu)

#multivariate t distribution
dS_2 = dS_2/chi_adjusted
dS_10 = dS_10/chi_adjusted
dg = dg/chi_adjusted

#convert to distribution function
dS_2 = t.cdf(dS_2,nu)
dS_10 = t.cdf(dS_10,nu)
dg = t.cdf(dg,nu)

#t copula gaussian marginal distribution
t_dS_2 = norm.ppf(dS_2)*std_dev_2
t_dS_10 = norm.ppf(dS_10)*std_dev_10
t_dg = norm.ppf(dg)*std_dev_g

#print results
print("t_2",t_dS_2)
print("t_10",t_dS_10)
print("t_g",t_dg)

#compute losses in portfolio value for both copulas
g_dV = -(w_2*G_dS_2 + w_10*G_dS_10 + w_g*G_dg)*10**8
t_dV = -(w_2*t_dS_2 + w_10*t_dS_10 + w_g*t_dg)*10**8
print("g_dV",g_dV)
print("t_dV",t_dV)

#compute VaR 99% for both copulas
VaR_G = np.percentile(g_dV,alpha[0]*100)
VaR_t = np.percentile(t_dV,alpha[0]*100)
print("99% VaR Gaussian copula",VaR_G)
print("99% VaR t copula",VaR_t)

#compute VaR 97.5% for both copulas to estimate ES
VaR_G_ES = np.percentile(g_dV,alpha[1]*100)
VaR_t_ES = np.percentile(t_dV,alpha[1]*100)
print("97.5% VaR Gaussian copula",VaR_G_ES)
print("97.5% VaR t copula",VaR_t_ES)

#compute ES 97.5% for both copulas
ES_G = g_dV[g_dV>VaR_G_ES].mean()
ES_t = t_dV[t_dV>VaR_t_ES].mean()
print("97.5% ES Gaussian copula",ES_G)
print("97.5% ES t copula",ES_t)

#compute worst quantile
worst_quantile_G_dS_10 = np.percentile(G_dS_10,alpha[2]*100)
worst_quantile_t_dS_10 = np.percentile(t_dS_10,alpha[2]*100)
print("worst quantile G dS_10", worst_quantile_G_dS_10)
print("worst quantile t dS_10", worst_quantile_t_dS_10)

#wrong way dependence effect
G_dg = np.where(G_dS_10>=worst_quantile_G_dS_10, G_dg + np.random.choice([0,-0.0003],size = N),G_dg)  
t_dg = np.where(t_dS_10>=worst_quantile_t_dS_10,t_dg + np.random.choice([0,-0.0003],size = N),t_dg)

#compute losses in portfolio value for both copulas with dependence effect
g_dV = -(w_2*G_dS_2 + w_10*G_dS_10 + w_g*G_dg)*10**8
t_dV = -(w_2*t_dS_2 + w_10*t_dS_10 + w_g*t_dg)*10**8

#compute VaR 99% for both copulas with dependence effect
VaR_G_dependence = np.percentile(g_dV,alpha[0]*100)
VaR_t_dependence = np.percentile(t_dV,alpha[0]*100)
print("99% VaR Gaussian copula with dependence effect",VaR_G_dependence)
print("99% VaR t copula with dependence effect",VaR_t_dependence)

#compute VaR 97.5% for both copulas to estimate ES with dependence effect
VaR_G_ES = np.percentile(g_dV,alpha[1]*100)
VaR_t_ES = np.percentile(t_dV,alpha[1]*100)
print("97.5% VaR Gaussian copula with dependence effect",VaR_G_ES)
print("97.5% VaR t copula with dependence effect",VaR_t_ES)

#compute ES 97.5% for both copulas with dependence effect
ES_G = g_dV[g_dV>VaR_G_ES].mean()
ES_t = t_dV[t_dV>VaR_t_ES].mean()
print("97.5% ES Gaussian copula with dependence effect",ES_G)
print("97.5% ES t copula with dependence effect",ES_t)
