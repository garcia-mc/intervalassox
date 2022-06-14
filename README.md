<p align="center">
  <a href="https://example.com/">
    <img src="https://github.com/garcia-mc/intervalassox/blob/main/results/baselinehazard.png" alt="Logo" width=300 height=300>
  </a>

  <h3 align="center">Neural interval-censored Cox regression with feature selection</h3>

  <p align="center">
    Carlos García, Marcos Matabuena and Michael R. Kosorok
    <br>
   
  </p>
</p>

# Abstract

The classical Cox model emerged in 1972 promoting breakthroughs in how patient prognosis is quantified using time-to-event analysis in biomedicine. One of the most useful characteristics of the
model for practitioners is the interpretability of the variables in the analysis. However, this comes at the price of introducing strong assumptions concerning the functional form of the regression model. To break this gap, this paper aims to exploit the explainability advantages of the classical
Cox model in the setting of interval-censoring using a new Lasso neural network that simultaneously selects the most relevant variables while quantifying non-linear relations
between predictors and survival times. The gain of the new method is illustrated empirically in an extensive simulation study with examples that involve linear and non-linear ground dependencies. We also demonstrate the performance of our strategy in the analysis of physiological, clinical and accelerometer data from the NHANES 2003-2006 waves to predict the effect of physical activity on the survival of patients. Our method outperforms the prior results in the literature that use the traditional Cox model.

# Launch in Euler cluster of ETH


```{bash}
scp -r garciac@euler.ethz.ch:intervalassox /intervalassox
ssh garciac@euler.ethz.ch

conda activate rpy2-env

cd intervalassox

rm -r results

mkdir results

bsub -n 24 -W 12:00 -o /output.txt 'python aaa.py'

cd ..

```
