# melissa-memory
documentation and tutorial on adding the capability of storing neural memory to a neural chatbot

There are 2 major innovations here:

1. Methods to quickly encode a memory at test time using few-shot meta-learning

2. Methods to keep an elastic vocabulary embedding 

# memory
This repository builds on the [chat-transformer](https://github.com/clam004/chat-transformer) chatbot
by adding a memory mechinism to the [transformer sequence2sequence chatbot](https://github.com/clam004/chat-transformer)
based on this [paper by Tsendsuren Munkhdalai et al](https://arxiv.org/pdf/1907.09720.pdf) called 
[Metalearned Neural Memory: Teaching neural networks how to remember](https://www.microsoft.com/en-us/research/blog/metalearned-neural-memory-teaching-neural-networks-how-to-remember/)

<img src = 'https://www.microsoft.com/en-us/research/uploads/prod/2019/12/MSR_NeuralMemory_V5_1400x788.gif' height=500 width=1000>

the high level gist of it is this: the controller in the diagram above is the sequence2sequence chatbot. In addition to taking a sequence as input and outputting a sequence as output, it also sends a signal to the memory network, a `key vector`, this is the input to a neural network telling it to retrieve a memory. That memory recall is returned to the chatbot in the form of the `value vector` which the chatbot can use to inform its next response. 
 
 # elastic vocab

 Typical seq2seq models have a fixed vocabulary, ie the number of words or tokens that can be used in the input and output are fixed. We have developed a method here for keeping a vocabulary is constantly growing and occasionally pruned. 

## How to Start

if you already have python 3.6 and virtual environments, create a python 3.6 virtual environment, here i used env36 for python3.6 but you can use anything

`python3 -m venv env`
or
`$ python3.6 -m venv env`
or
`virtualenv --python=/usr/bin/python3.6 env`

if python3.6 is your default version, then when you type `python` into your terminal then it should say python version 3.6.x, and for you making the virtual environment is as simple as 

`$ python -m venv env`

otherwise it is simple to get python3.6 and virtual environments

[how to install Python 3.6 on ubuntu](http://ubuntuhandbook.org/index.php/2017/07/install-python-3-6-1-in-ubuntu-16-04-lts/)

install [virtual environment](https://towardsdatascience.com/virtual-environments-104c62d48c54) then 

[how to specify the Python executable you want to use](https://stackoverflow.com/questions/1534210/use-different-python-version-with-virtualenv)


`sudo add-apt-repository ppa:jonathonf/python-3.6`

`sudo apt-get update`

`sudo apt-get install python3.6`

`virtualenv --python=/usr/bin/python3.6 env36`


When you want to run the code activate the virtual environment inside the same folder as your environment env using 

`$ source env/bin/activate`

install dependencies

`$ pip3 install -r requirements.txt`

[even with virtual environments, some troubleshoot might be needed](https://github.com/tensorflow/tensorflow/issues/559)

[with enough google searches you can find an answer for almost any problem](https://stackoverflow.com/questions/45912674/attributeerror-module-numpy-core-multiarray-has-no-attribute-einsum)

save new dependences to requirements

`$ pip3 freeze > requirements.txt`

You can deactivate the virtual environment using the following command in your terminal:

`$ deactivate`

## More Tips and Tricks

if you get a `ImportError: No module named` while at the same time in your Terminal you get 

`pip3 install import-ipynb`


`Requirement already satisfied: import-ipynb in /path/to/env/lib/python3.6/site-packages (0.1.3)`


This can be fixed by providing your python interpreter with the path-to-your-module,the path 

`import sys`

`sys.path.append('/path/to/env/lib/python3.6/site-packages')` 

you can find this path listed here

`$ python3 -m site`
