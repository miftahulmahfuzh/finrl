import numpy as np
def df_to_array(df, tech_indicator_list):
    df = df.copy()
    unique_ticker = df.tic.unique()
    if_first_time = True
    for tic in unique_ticker:
        if if_first_time:
            price_array = df[df.tic == tic][["close"]].values
            tech_array = df[df.tic == tic][tech_indicator_list].values
            turbulence_array = df[df.tic == tic]["turbulence"].values
            if_first_time = False
        else:
            price_array = np.hstack(
                [price_array, df[df.tic == tic][["close"]].values]
            )
            tech_array = np.hstack(
                [tech_array, df[df.tic == tic][tech_indicator_list].values]
            )
    tech_nan_positions = np.isnan(tech_array)
    tech_array[tech_nan_positions] = 0
    tech_inf_positions = np.isinf(tech_array)
    tech_array[tech_inf_positions] = 0
    return price_array, tech_array, turbulence_array

price_array, tech_array, turbulence_array = df_to_array(data)

env_config = {
    "price_array": price_array,
    "tech_array": tech_array,
    "turbulence_array": turbulence_array,
    "if_train": True,
}
env_instance = env(config=env_config)

# read parameters
cwd = kwargs.get("cwd", "./" + str(model_name))

if drl_lib == "elegantrl":
    DRLAgent_erl = DRLAgent
    break_step = kwargs.get("break_step", 1e6)
    erl_params = kwargs.get("erl_params")
    agent = DRLAgent(
        env=env,
        price_array=price_array,
        tech_array=tech_array,
        turbulence_array=turbulence_array,
    )
    model = agent.get_model(model_name, model_kwargs=erl_params)
    trained_model = agent.train_model(
        model=model, cwd=cwd, total_timesteps=break_step
    )

################################################################
env_config = {
    "price_array": price_array,
    "tech_array": tech_array,
    "turbulence_array": turbulence_array,
    "if_train": False,
}
env_instance = env(config=env_config)

# load elegantrl needs state dim, action dim and net dim
net_dimension = kwargs.get("net_dimension", 2**7)
cwd = kwargs.get("cwd", "./" + str(model_name))
print("price_array: ", len(price_array))

if drl_lib == "elegantrl":
    DRLAgent_erl = DRLAgent
    episode_total_assets = DRLAgent_erl.DRL_prediction(
        model_name=model_name,
        cwd=cwd,
        net_dimension=net_dimension,
        environment=env_instance,
    )
    return episode_total_assets

