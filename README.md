# FlexRLHF: A Flexible Placement and Parallelism Framework for Efficient RLHF Training
We present our demo for the paper: "FlexRLHF: A Flexible Placement and Parallelism Framework for Efficient RLHF Training." 
Due to integration with internal systems and confidentiality constraints, we are only able to provide a demo version of the code, which illustrates the core concepts and functionality of FlexRLHF.

# 🏃 How to train RLHF in FlexRLHF framework
## Run Interleaving Strategy
```shell
sh run_interleaving.sh
```

## Run Flattening Strategy
```shell
sh run_flattening.sh
```
# 🌲Code structure
```
alignment
├── __init__.py
├── api
│   ├── __init__.py
│   ├── rlhf
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── model_provider.py
│   │   └── rlhf_trainner.py
│   └── utils
│       ├── __init__.py
│       └── dist_util.py
├── app
│   ├── __init__.py
│   └── util
│       ├── base_logger.py
│       ├── logger.py
│       └── py_utils.py
├── conf
│   ├── flattening.yaml
│   └── interleaving.yaml
├── examples
│   └── app_main.py
├── rlhf
│   ├── data
│   │   ├── __init__.py
│   │   ├── data_utils.py
│   │   ├── default_data_impl.py
│   │   └── raw_datasets.py
│   ├── distributed
│   │   ├── __init__.py
│   │   ├── distributed_rlhf_engine.py
│   │   └── model_placement.py
│   ├── model
│   │   ├── __init__.py
│   │   ├── default_model_impl.py
│   │   ├── model_utils.py
│   │   └── reward_model.py
│   ├── module
│   │   ├── __init__.py
│   │   ├── ac_none_share_module.py
│   │   ├── rlhf_module.py
│   │   └── util.py
│   ├── trainner
│   │   ├── __init__.py
│   │   ├── app_ds_rlhf_engine.py
│   │   ├── exec_engine.py
│   │   └── mixed_model.py
│   └── utils
│       ├── __init__.py
│       ├── ds_utils.py
│       └── utils.py
└── util
    ├── __init__.py
    └── global_vars.py
```