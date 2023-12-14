import torch
import numpy as np
import pandas as pd
import shutil

from darts.models import NHiTSModel, TiDEModel
from darts.datasets import AusBeerDataset, WeatherDataset
from darts.dataprocessing.transformers.scaler import Scaler
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from darts.metrics import mae, mse
import matplotlib.pyplot as plt
from darts.metrics import mae, mse
import pandas as pd



optimizer_kwargs = {
    "lr": 1e-3,
}

# PyTorch Lightning Trainer arguments
pl_trainer_kwargs = {
    "gradient_clip_val": 1,
    "max_epochs": 200,
    "accelerator": "auto",
    "callbacks": [],
}

# learning rate scheduler
lr_scheduler_cls = torch.optim.lr_scheduler.ExponentialLR
lr_scheduler_kwargs = {
    "gamma": 0.999,
}

# early stopping (needs to be reset for each model later on)
# this setting stops training once the the validation loss has not decreased by more than 1e-3 for 10 epochs
early_stopping_args = {
    "monitor": "val_loss",
    "patience": 10,
    "min_delta": 1e-3,
    "mode": "min",
}

#
common_model_args = {
    "input_chunk_length": 12,  # lookback window
    "output_chunk_length": 12,  # forecast/lookahead window
    "optimizer_kwargs": optimizer_kwargs,
    "pl_trainer_kwargs": pl_trainer_kwargs,
    "lr_scheduler_cls": lr_scheduler_cls,
    "lr_scheduler_kwargs": lr_scheduler_kwargs,
    "likelihood": None,  # use a likelihood for probabilistic forecasts
    "save_checkpoints": True,  # checkpoint to retrieve the best performing model state,
    "force_reset": True,
    "batch_size": 256,
    "random_state": 42,
}


series = AusBeerDataset().load()

train, temp = series.split_after(0.6)
val, test = temp.split_after(0.5)

scaler = Scaler()  # default uses sklearn's MinMaxScaler
train = scaler.fit_transform(train)
val = scaler.transform(val)
test = scaler.transform(test)

# create the models
model_nhits = NHiTSModel(**common_model_args, model_name="hi")


model_tide = TiDEModel(
    **common_model_args, use_reversible_instance_norm=True, model_name="tide1"
)

models = {
    "NHiTS": model_nhits,
    "TiDE": model_tide,
}

# train the models and load the model from its best state/checkpoint
for name, model in models.items():

    # early stopping needs to get reset for each model
    pl_trainer_kwargs["callbacks"] = [
        EarlyStopping(
            **early_stopping_args,
        )
    ]

    model.fit(
        series=train,
        val_series=val,
        verbose=False,
    )
    # load from checkpoint returns a new model object, we store it in the models dict
    models[name] = model.load_from_checkpoint(model_name=model.model_name, best=True)


# Assuming 'common_model_args' and 'test' are defined elsewhere in your code
pred_steps = common_model_args["output_chunk_length"] * 2
pred_input = test[:-pred_steps]

fig, ax = plt.subplots(figsize=(15, 5))
pred_input.plot(label="Input", color="blue") # Color for input
test[-pred_steps:].plot(label="Ground Truth", color="green", ax=ax) # Color for ground truth

result_accumulator = {}
model_colors = ["red", "orange", "purple", "brown"]  # Define a list of colors for the models

# Predict with each model and compute/store the metrics against the test sets
for i, (model_name, model) in enumerate(models.items()):
    pred_series = model.predict(n=pred_steps, series=pred_input)
    color = model_colors[i % len(model_colors)]  # Cycle through model_colors
    pred_series.plot(label=model_name, color=color, ax=ax)  # Use the chosen color

    result_accumulator[model_name] = {
        "mae": mae(test, pred_series),
        "mse": mse(test, pred_series),
    }

ax.legend()  # Show the legend
ax.set_title("Time Series Prediction Comparison")  # Set the title
plt.show()  # Display the plot


# Assuming 'result_accumulator' is defined elsewhere in your code
results_df = pd.DataFrame.from_dict(result_accumulator, orient="index")

# Define a list of colors for the bars
bar_colors = ["orange", "green"]

# Plot the bar chart with the specified colors
results_df.plot.bar(color=bar_colors)

# Optionally, you can add titles and labels for clarity
plt.title("Model Performance Comparison")
plt.ylabel("Metric Value")
plt.xlabel("Models")
plt.show()

