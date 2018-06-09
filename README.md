# dogs & cats classification
# Dataset
## Part 1 Create folders and copy images
Dataset is from a kaggle competition [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data)  
Part1_Create New folder & Copy images.py is to seperate the training data by lables.
    The image stracutre is like this<br>
		├─test
		│  └─test
		│          1.jpg
		│          10.jpg
		│          100.jpg
		│          1000.jpg
		│          10000.jpg
		│          10001.jpg

		│          
		├─train
		│  ├─cats
		│  │      cat.0.jpg
		│  │      cat.1.jpg
		│  │      cat.2.jpg
		│  │      cat.3.jpg
		│  │      cat.4.jpg

		│  │      
		│  └─dogs
		│          dog.0.jpg
		│          dog.1.jpg
		│          dog.2.jpg
		│          dog.3.jpg
		│          dog.4.jpg

		│          
		└─validation
			├─cats
			│      cat.10.jpg
			│      cat.11.jpg
			│      cat.12.jpg
			│      cat.13.jpg

			│      
			└─dogs
					dog.10.jpg
					dog.11.jpg
					dog.12.jpg


# Structure of model
![structure of model](https://github.com/silentwhaleluo/dogs---cats-classification-/blob/master/model%20structure.png?raw=true)
## Part 2 Extract features
Using pretraining models to extract features and then flatten them and combine them together.
## Part 3 Train top model and save results
Using fully connected neural network to train the model
