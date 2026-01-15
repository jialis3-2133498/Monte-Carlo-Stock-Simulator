# Monte-Carlo-Stock-Simulator
This project aims to estimate the distribution of a stock's future price/return over a 5-year horizon and summarize risk.
## Parameters
1. The target stock is Amazon's stock.
2. The estimation range is from 01/01/2020 to 01/01/2025. The simulation horizon is 5 years into the future.
3. Random Seed will be set as 422.
4. The number of simulation paths is 1000.
5. Forecast horizon in trading days is 1260 trading days.

   $5 \times 252 = 1260$ trading days

7. Time step $\Delta t$ is $1/252$
## Expected Output
1. It will produce metrics/tables about the expected return, median return, probability of loss, expected end price, risk matrix, and median end price
2. It will also generate plots to visualize the distribution of the stock's price.
## Model Approach/Assumptions
1. We will use adjusted close prices in this project.
2. We model daily log returns as i.i.d. Normal:

   $r_t = \ln(S_t/S_{t-1})$

   and assume

   $r_t \sim \mathcal{N}(\mu, \sigma^2)$ (i.i.d.).

   Therefore, under GBM, the future price $S_T$ is lognormally distributed.

4. We assume trading days per year are 252 days.
5. We will use MLE for the model's estimates.
## Payoff in this project
It is the buy-and-hold payoff, and we define the cumulative log return over the horizon as:

  $g(S_T) = \ln(S_T / S_0)$

The expected payoff is under the model.
## Notes about using this model
1. Command line usage
