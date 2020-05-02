## Imports and set-up
import datetime as dt
import os
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import seaborn as sns
import sklearn
import torch
from IPython.core.display import HTML, display
from jax import lax, random, vmap
from jax.scipy.special import logsumexp
from numpyro import handlers, optim
from numpyro.infer import ELBO, MCMC, NUTS, SVI

## Normalize dataset
from sklearn.preprocessing import StandardScaler

from tools import *

numpyro.set_host_device_count(4)


## Function definitions
@logger
def select_meter(df, meter=0):  # Electricity = 0
    return df[df.meter == 0].drop(columns=["meter"])


@logger
def drop_outliers(df, threshold=5000):  # does this threshold makes sense?
    return df[df.meter_reading < threshold]


@logger
def to_datetime(df):
    df.timestamp = pd.to_datetime(df.timestamp)
    return df


def compute_error(trues, predicted):
    corr = np.corrcoef(predicted, trues)[0, 1]
    mae = np.mean(np.abs(predicted - trues))
    rae = np.sum(np.abs(predicted - trues)) / np.sum(
        np.abs(trues - np.mean(trues)))
    rmse = np.sqrt(np.mean((predicted - trues)**2))
    r2 = max(
        0, 1 - np.sum((trues - predicted)**2) / np.sum(
            (trues - np.mean(trues))**2))
    return corr, mae, rae, rmse, r2


## Load Dataset
df_meta_raw = pd.read_csv("2_Data/building_metadata.csv")
df_raw = pd.read_csv("2_Data/train.csv")
df_weather_raw = pd.read_csv("2_Data/weather_train.csv")

df = (
    df_raw.pipe(start_pipeline).pipe(reduce_mem_usage).pipe(
        select_meter)  # only electricity
    .pipe(drop_outliers
          )  # check this threshold, we haven't done any visual check
    .pipe(to_datetime))
df_meta = df_meta_raw.pipe(start_pipeline).pipe(reduce_mem_usage)
df_weather = (df_weather_raw.pipe(start_pipeline).pipe(reduce_mem_usage).pipe(
    to_datetime))

## Filter raw datasets to include only households
df_meta = df_meta[df_meta.primary_use.eq("Lodging/residential")]
hh_id = df_meta.building_id.unique()
df = df[df.building_id.isin(hh_id)]
hh_id2 = df.building_id.unique(
)  # repeat because 2 ids are missing from df_meta
df_meta = df_meta[df_meta.building_id.isin(hh_id2)]
hh_site_id = df_meta.site_id.unique()
df_weather = df_weather[df_weather.site_id.isin(hh_site_id)]

## Keep weather feautures if <10% is nan
df_weather = df_weather.loc[:, df_weather.isna().mean() < 0.2]
df_weather = df_weather.dropna()

## keep one area for now
df_weather = df_weather[df_weather.site_id == 0]
df_meta = df_meta[df_meta.site_id == 0]
df = df[df.building_id.isin(df_meta.building_id)]

## join datasets
df = df.set_index("building_id").join(df_meta.set_index("building_id"),
                                      how="inner")
df = df.reset_index()  # to keep building_id
df = df.set_index("timestamp").merge(df_weather.set_index("timestamp"),
                                     on=["site_id", "timestamp"],
                                     how="left")

## keep one household for now
df = df[df.building_id == 100]
df = df[["wind_speed", "air_temperature", "dew_temperature",
         "meter_reading"]].dropna()

## Create Dataset
X = df[["wind_speed", "air_temperature", "dew_temperature"]].values
y = df.meter_reading.values
## In this particular example before idx 4000 there are no measurements
X = X[4000:, :]
y = y[4000:]
## Normalize
x_scaler = StandardScaler().fit(X)
X = x_scaler.transform(X)
y_m = jnp.mean(y)
y_std = jnp.std(y)
y = (y - y_m) / y_std
## Undersample
X = X[::12]
y = y[::12]


## Model Functions
def f(carry, inputs):
    x_prev, noise_t = inputs
    beta, eta, z_prev, tau = carry
    z_t = beta * z_prev + jnp.dot(x_prev, eta) + tau * noise_t
    z_prev = z_t
    return (beta, eta, z_prev, tau), z_t


def model(T, T_forecast, X, obs=None):
    """ 
    Define priors over delta, tau, noises, sigma, z_prev, eta
    """
    tau = numpyro.sample("tau", dist.HalfCauchy(10.0))
    noises = numpyro.sample(
        "noises",
        dist.Normal(jnp.zeros(X.shape[0]), 5.0 * jnp.ones(X.shape[0])),
    )
    sigma = numpyro.sample("sigma", dist.HalfCauchy(scale=5.0))
    delta = numpyro.sample("delta", dist.Normal(0, 5.0))

    z_prev = numpyro.sample("z_prev", dist.Normal(0, 3.0))

    eta = numpyro.sample(
        "eta",
        dist.Normal(jnp.zeros(X.shape[1]), 5.0 * jnp.ones(X.shape[1])),
    )
    """ Propagate the dynamics forward using jax.lax.scan
    """
    carry = (delta, eta, z_prev, tau)
    z_collection = [z_prev]
    carry, zs_exp = lax.scan(
        f=f,
        init=carry,
        xs=(X, noises),
    )
    z_collection = jnp.concatenate((jnp.array(z_collection), zs_exp), axis=0)
    """ Sample the observed_y (y_obs) and predicted_y (y_pred) - note that you don't need a pyro.plate!
    """
    numpyro.sample("y_obs", dist.Normal(z_collection[:T], sigma), obs=obs)
    numpyro.sample("y_pred", dist.Normal(z_collection[T:], sigma), obs=None)

    return z_collection


## Test model
# TODO check problem with jax scan length
rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)

kernel = NUTS(model, max_tree_depth=8, step_size=5e-3)
mcmc = MCMC(kernel, num_warmup=2000, num_samples=2000, num_chains=4)
##
n_test = 50
mcmc.run(random.PRNGKey(2),
         T=len(X) - n_test,
         T_forecast=n_test,
         X=X,
         obs=y[:-n_test])
mcmc.print_summary()
##
samples = mcmc.get_samples()
## Generate Predictions
hmc_samples = {k: v for k, v in mcmc.get_samples().items()}
q = hmc_samples["y_pred"]
q_mean = q.mean(axis=0).reshape(-1, )
q_std = q.std(axis=0).reshape(-1, )
y_pred_025 = q_mean - 1.96 * q_std
y_pred_975 = q_mean + 1.96 * q_std
y_pred_mean = q_mean

## Plot results
# plt.plot(ix_train[-20:], y_train[-20:, 0], "b-")
# plt.plot(ix_test, y[:], "bx")
ix_train = range(len(y) - n_test)
ix_test = range(len(y) - n_test - 1, len(y))
plt.figure(figsize=(10, 6))
plt.plot(ix_train, y[:-n_test])
plt.plot(ix_test, y_pred_mean, "r-")
plt.plot(ix_test, y_pred_025, "r--")
plt.plot(ix_test, y_pred_975, "r--")
plt.fill_between(ix_test, y_pred_025, y_pred_975, alpha=0.3)
plt.legend([
    "true (train)",
    "true (test)",
    "forecast",
    "forecast + stddev",
    "forecast - stddev",
])
plt.show()
