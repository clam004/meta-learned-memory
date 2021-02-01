# melissa-memory
documentation and tutorial on adding the capability of storing neural memory to a neural chatbot
This repository builds on the [chat-transformer](https://github.com/clam004/chat-transformer) chatbot
by adding a memory mechinism to the [transformer sequence2sequence chatbot](https://github.com/clam004/chat-transformer)
based on this [paper by Tsendsuren Munkhdalai et al](https://arxiv.org/pdf/1907.09720.pdf) called 
[Metalearned Neural Memory: Teaching neural networks how to remember](https://www.microsoft.com/en-us/research/blog/metalearned-neural-memory-teaching-neural-networks-how-to-remember/)

<img src = 'https://www.microsoft.com/en-us/research/uploads/prod/2019/12/MSR_NeuralMemory_V5_1400x788.gif' height=500 width=1000>

the high level gist of it is this: the controller in the diagram above is the sequence2sequence chatbot. In addition to taking a sequence as input and outputting a sequence as output, it also sends a signal to the memory network, a `key vector`, this is the input to a neural network telling it to retrieve a memory. That memory recall is returned to the chatbot in the form of the `value vector` which the chatbot can use to inform its next response. 

The innovation in this approch is the methods demostrated here on how to quickly encode a memory at test time using few-shot meta-learning 
