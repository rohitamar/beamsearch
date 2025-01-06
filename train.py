import torch 
import torch.optim as optim
import torch.nn as nn

torch.manual_seed(6969)

with open('./dataset/data.txt') as f:
    data = f.read()
    vocab = sorted(set(data))
    ctoi = {x:i for i, x in enumerate(vocab)}
    itoc = {i:x for i, x in enumerate(vocab)}
    vocab_size = len(vocab)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Training on: ", device)

def encode(s):
    return [ctoi[x] for x in list(s)]

def decode(s):
    return "".join(itoc[x] for x in s)

train_size = int(0.7 * len(data))
val_size = int(0.2 * len(data))
train_data, val_data, test_data = data[:train_size], data[train_size:train_size+val_size], data[train_size+val_size:]
train_data, val_data, test_data = encode(train_data), encode(val_data), encode(test_data)

def get_batch(block_size, batch_size, split = "train"):
    if split == "train":
      data = train_data
    else:
      data = val_data if split == "val" else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.tensor(data[i:i+block_size]) for i in ix])
    y = torch.stack([torch.tensor(data[i+1:i+block_size+1]) for i in ix])
    return x.to(device), y.to(device)

# assumes that scheduler being used is ReduceLROnPlateau (if used) 
def train_loop(model, criterion, schedule, configs):

    block_size = configs['block_size']
    batch_size = configs['batch_size']
    steps = configs['steps']
    eval = configs['eval']
    test_every = configs['test_every']
    init_lr = configs['init_lr']

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = init_lr)
    if schedule:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    train_loss, val_loss, test_loss = [], [], []

    def get_loss(split):
        x, y = get_batch(block_size, batch_size, split)
        preds = model(x)
        target = y.reshape(batch_size * block_size)
        preds = preds.reshape(batch_size * block_size, -1)

        loss = criterion(preds, target)
        return loss

    for i in range(steps):
        loss = get_loss("train")
        train_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % test_every == 0:
            with torch.no_grad():
                vl = 0.0
                for _ in range(eval):
                    loss = get_loss("val")
                    vl += loss.item()
                vl = vl / 200
                val_loss.append(vl)

                tl = 0.0
                for _ in range(eval):
                    loss = get_loss("test")
                    tl += loss.item()
                tl = tl / 200
                test_loss.append(tl)

                if schedule:
                    scheduler.step(tl)
                
                print(f"Step [{i}] Train: {train_loss[-1]} Val: {val_loss[-1]} Test: {test_loss[-1]}")
    
    return {
        "train": train_loss,
        "val": val_loss,
        "test": test_loss
    }

def train_loop_nplm(model, criterion, schedule, configs):

    block_size = configs['block_size']
    batch_size = configs['batch_size']
    steps = configs['steps']
    eval = configs['eval']
    test_every = configs['test_every']
    init_lr = configs['init_lr']

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = init_lr)
    if schedule:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    train_loss, val_loss, test_loss = [], [], []

    def get_loss(split):
        x, _ = get_batch(block_size, batch_size, split)
        preds = model(x[:, :-1])
        target = x[:, -1]


        loss = criterion(preds, target)
        return loss

    for i in range(steps):
        loss = get_loss("train")
        train_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % test_every == 0:
            with torch.no_grad():
                vl = 0.0
                for _ in range(eval):
                    loss = get_loss("val")
                    vl += loss.item()
                vl = vl / 200
                val_loss.append(vl)

                tl = 0.0
                for _ in range(eval):
                    loss = get_loss("test")
                    tl += loss.item()
                tl = tl / 200
                test_loss.append(tl)

                if schedule:
                    scheduler.step(tl)
                
                print(f"Step [{i}] Train: {train_loss[-1]} Val: {val_loss[-1]} Test: {test_loss[-1]}")
    
    return {
        "train": train_loss,
        "val": val_loss,
        "test": test_loss
    }

from models import NPLM, RNN, TransformerModel

rnn_model = RNN(vocab_size=vocab_size, 
                embed_size=64, 
                hidden_size=64
            ).to(device)

rnn_loss = train_loop(
    rnn_model, 
    nn.CrossEntropyLoss(), 
    False, 
    {
        "block_size": 32, 
        "batch_size": 16, 
        "steps": 30000, 
        "eval": 200, 
        "test_every": 1000,
        "init_lr": 1e-3
    }
)

transformer_model = TransformerModel(
                        vocab_size=vocab_size, 
                        d_model=64
                    ).to(device)

transformer_loss = train_loop(
    transformer_model, 
    nn.NLLLoss(), 
    True, 
    {
        "block_size": 32, 
        "batch_size": 32, 
        "steps": 30000, 
        "eval": 200, 
        "test_every": 1000,
        "init_lr": 1e-3
    }
)

nplm_model = NPLM(vocab_size=vocab_size, 
                  embed_size=64, 
                  hidden_size=64, 
                  block_size=32
             ).to(device)

nlpm_loss = train_loop_nplm(
    nplm_model, 
    nn.CrossEntropyLoss(),
    False,
    {
        "block_size": 32, 
        "batch_size": 32, 
        "steps": 30000, 
        "eval": 200, 
        "test_every": 1000,
        "init_lr": 1e-3
    }
)

torch.save(rnn_model.state_dict(), "./saved_models/rnn.pt")
torch.save(transformer_model.state_dict(), "./saved_models/transformer.pt")
torch.save(nplm_model.state_dict(), "./saved_models/nplm.pt")
