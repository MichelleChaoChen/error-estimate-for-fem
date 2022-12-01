# error-estimate-for-fem
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