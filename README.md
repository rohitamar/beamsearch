# Sampling different models

Comparing different sampling techniques on N-grams, NPLM, LSTMs, and Transformers.  

## Number of Parameters and Hyperparameters

For fun, here's the number of parameters: 
| **Model**      | **Number of Parameters**  |
|----------------|------------|
| Transformer    | 1,132,993    |
| RNN            | 41,665      |
| NPLM           | 319,545     |

All models were trained on a block_size of 32, batch_size of 32, learning_rate of 1e-3, and 30,000 steps. For the NPLM, I used a hidden layer of size 300. For the RNN, I used a hidden layer of size 64. For the Transformer, I used n_layers = 4, d_model = 64, n_head = 4 and dropout of 0.1. 

## Outputs
Transformer
```
 more.
Cannot the Citizen:
What come, dock's heart; not care too your son Edward's pain.

KING RICHARD III:
Death, yet up, and that I dear still nor than comes!

First Servingman:
Why had, and farewell; then I  would steep him.

GLOUCESTER:
Have this fight, to me gracious son:
And sick, and the maste
```

RNN
```
 hood war me;
You clound and to lader.

BUCKID:
Me, a pack his chipts and behin both to.

KING EDWARD IV:
Gook now shatt.
O well.
Seeds
A deaked no hour. I have days,
And course proved onror, as the gared everble bricking sties than be have Tyseng chand shalt soums;
And some wrong of presence some tu
```

NPLM (context: ``Then? and I crown, with land is it so``)
```
Then? and I crown, with land is it sounta? 
Ondhf'd, so dey of oir of it'oun nane agn boad, and hocast,
We cakell the gignne
SEspald to meland wour outhse?-is fole to hatcaon.

HORBIUBOLIO:
Atlo dowe that haut ary acers;
Teta sold Ruclafoft wo holl with hed mreak! will twh,
Thiss the quick both rick,
And batiche lorn you pad inince!
Th
```
