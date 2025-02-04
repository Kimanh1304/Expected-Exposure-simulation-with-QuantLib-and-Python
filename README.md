# Expected-Exposure-simulation-with-QuantLib-and-Python

This code shows how to use the Python bindings of the QuantLib library to calculate the expected exposure (EE) for a netting set of interest rate swaps. 

Using a forward Monte Carlo Simulation to generate future market scenarios out of one-factor gaussian short rate model and evaluate the NPV of all swaps in the netting set under each scenario.

## Define a time grid. 
On each date/time in our grid we want to calculate the expected exposure. For each date in our time grid we will simulate N states of the market and for each of these states we will calculate the NPV all of instruments in our portfolio / netting set. This results in N x (size of the netting set) simulated paths of NPVs.

The total number of NPV evaluations is (size of time grid) x (size of portfolio) x N. For a big portfolio and a very dense time grid it can be very time consuming task even if the single pricing is done pretty fast.

For simplicity we restrict the portfolio to plain vanilla interest rate swaps in one currency. Further we assume that we live in a “single curve” world. We will use the same yield curve for discounting and forwarding. No spreads between the different tenor curves neither CSA discounting are taken into account. For the swap pricing we will need future states of the yield curve. In our setup we assume the the development of the yield curve follow an one factor Hull-White model. At the moment we make no assumption on how it is calibrated and assume its already calibrated. In our setting we will simulate N paths of the short rate following the Hull-White dynamics. At each time on each path the yield curve depend only on the state of our short rate process. We will use QuantLib functionalities to simulate the market states and perform the swap pricing on each path.

## Setup of the market state at time zero (today)

## Setup portfolio / netting set
Our netting set consists of two swaps, one receiver and one payer swap. Both swaps differ also in notional and time to maturity. Finally we create a pricing engine and link each swap in our portfolio with it.

## Monte-Carlo-Simulation of the “market”
We select a weekly time grid, including all fixing days of the portfolio. To generate the future yield curves we are using the GSR model and process of the QuantLib.

The GSR model allows the mean reversion and the volatility to be piecewise constant. In our case here both parameter are set constant. Given a time $t_0$ and state $x(t_0)$ of the process we know the conditional transition density for $x(t_1)$ for $t_1 > t_0$. Therefore we don’t need to discretize the process between the evaluation dates. As a random number generator we are using the Mersenne Twister.

We also save the zero bonds prices on each scenario for a set of maturities (6M, 1Y,…,10Y). We use this prices as discount factors for our scenario yield curve.
## Pricing on path & netting
On each date t and on each path p we will evaluate the netting set. First we build a new yield curve using the scenario discount factors from the step before.

After relinking the yield termstructure handle is the revaluation of the portfolio is straight forward. We just need to take fixing dates into account and store the fixings otherwise the pricing will fail.

## Calculation EE and PFE
After populating the cube of fair values (1st dimension is simulation number, 2nd dimension is the time and 3rd dimension is the deal) we can calculate the expected exposure and the potential future exposure.

