# Error Estimation for Finite Element Method

[![Open Documentation](https://img.shields.io/static/v1?label=Documentation&message=Read&color=brightgreen&logo=readthedocs)](https://michellechaochen.github.io/error-estimate-for-fem/)
[![Open in Dev Containers](https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/MichelleChaoChen/error-estimate-for-fem)


## Setting up Your Environment 


## Adaptive Mesh Refinement



## Customizing the Problem 


### Generating Data


### Training a Model 
To train your own model:
```
python train.py --data "path to training data"
```
The trained model will be stored in the **src/models** folder

To run the code 
```
python main.py --model "path to your model" --epochs num_epochs
```
Where your model must be of the extension .ckpt

### Running Adaptive Mesh Refinement (AMR)


### Evaluating AMR Results
