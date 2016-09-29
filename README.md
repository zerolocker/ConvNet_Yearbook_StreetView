# ConvNet_Yearbook_StreetView


### VGG-16/VGG-19 Model
The [tensorflow-vgg](https://github.com/machrisaa/tensorflow-vgg) repo is contained in this repo as a ["fake submodule"](http://debuggable.com/posts/git-fake-submodules:4b563ee4-f3cc-4061-967e-0e48cbdd56cb)

### Usage

1. Clone this repository
  ```bash
  git clone this_repository
  cd this_repository
  ```

2. Download yearbook dataset
  ```
  wget http://www.cs.utexas.edu/~philkr/cs395t/yearbook_trainval.tar.gz
  tar -xzf yearbook_trainval.tar.gz 
  ```

3. create a small subset of yearbook dataset
  ```
  ipython subset_dataset.py
  ```

4. test if input_pipeline.py works correctly
  ```
  ipython input_pipeline.py
  ```
  input_pipeline.py is a file used by almost all other files to create an "input pipeline" in the tensorflow graph. (Graph is a concept in tensorflow.https://www.tensorflow.org/versions/r0.10/api_docs/python/framework.html) 
  
5. choose whatever model you want to work on:

  **`ipython runvgg.py`**: train VGG-19.  
  You may want to tune the parameter by running `python runvgg.py -learning_rate 1e-4 -eps 1e-8` to get good results. Note: if you want to specify parameters, you cannot use `ipython runvgg.py`, you have to use `python runvgg.py`. I don't know why this won't work for ipython...
  
  **`ipython fully_connected.py`**: train a 2 hidden layer network.  
  
  **`ipython tinyfull.py`**: train a softmax classifier.  
  i.e. a tiny fully connected network with no hidden layers.  
  `tinyfull.py` is designed to be as short and simple as possible so that I can test out new code / debugging a Tensorflow problem. So if you want to know the structure of the other files, I recommend reading the code in `tinyfull.py` first.
  
