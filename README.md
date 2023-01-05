# scPDB

## Requirements
* Anaconda
* Cuda 11.7.1

## How to Run
* Build the conda environments in /envs/ob_env.yml and /envs/pytorch_env.yml (the clustering environment is currently defunct).
* Copy provided data and unzip in the main directory
### Training
The following command can be used to train the model:
```
python3 train.py -s <split>
```
where ```<split>``` can be one of ["val", "coach420_mlig", "holo4k_mlig"]
Example slurm scripts for reference can be found in ```./bridges_slurm_scripts/train_scripts.```

### Performing Inference
Inference can be performed with the following command:
```
python3 infer_test_set.py <split> <path_to_trained_model>
```
Example slurm scripts can be found in ```./bridges_slurm_scripts/test_inference_slurm_scripts/[ job_test_val.sh | job_test_coach420_mlig.sh | job_test_holo4k_mlig.sh ]```


### Calculating Metrics
Once test inference has completed, common metrics can be calculated using:
```
python3 site_metrics.py <split> <path_to_trained_model>
```
Example slurm scripts can be found in ```./bridges_slurm_scripts/test_inference_slurm_scripts/[ job_test_site_metrics_val.sh | job_test_site_metrics_coach420_mlig.sh | job_test_site_metrics_holo4k_mlig.sh ]```
