import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
print (torch.float)
""" load dataset """
train_x_data = torch.tensor([
    [ 10, 2, 5 ],
    [ 5, 0, 40 ],
    [ 8, 1, 40 ],
    [ 9, 0, 40 ],
    [ 0, 0, 140],
    [ 1, 0, 50 ]
    ],  dtype=torch.float)

train_y_data = torch.tensor([
    [1],[0],[1],[1],[0], [0]
], dtype=torch.float)

class Model (nn.Module) :
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(3, 1)
        self.loss = nn.BCELoss(size_average=True)

model = Model()

optimzer = optim.Adam(model.parameters(), lr=.01)

for epoch in range(1000):     
    y_pred = F.sigmoid(model.linear(train_x_data)) 
    loss = model.loss(y_pred, train_y_data)   
    print("loss",loss.item())
    optimzer.zero_grad()
    loss.backward()
    optimzer.step


input_x = torch.tensor([[20,2,30]], dtype=torch.float) 


print(  F.sigmoid(model.linear(input_x)) )