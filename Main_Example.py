import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score

from MCVN import MCVN
from MyDataLoader import MyDataLoader
from Trainer import Trainer

if __name__ == '__main__':
    '''
    An example of training MCVN is now provided, and it is worth noting 
    that the parameters in it are used for demonstration purposes and 
    differ from the parameter settings in the paper, and that there 
    are many default parameters used, which can be modified in ways 
    found in the specific code to obtain better training results.
    
    There are only 10 samples of data in the repository, and all 
    data can be obtained from the corresponding author
    '''
    # init model
    model = MCVN(16)
    # get dataset
    dataloader = MyDataLoader(data_path='./Data/NIMS/NIMS_Fatigue.csv', img_path='./Data/NIMS/Images.csv', targets=['Fatigue', 'Tensile', 'Fracture', 'Hardness'])
    Train, Valid, Test = dataloader.get_dataset(normal_feature=False, normal_target=False)
    train_data = Train['Fatigue']
    valid_data = Valid['Fatigue']
    test_data = Test['Fatigue']
    # init training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)
    # training
    trainer = Trainer(model=model, train_data = train_data,
                 valid_data = valid_data, criterion=criterion, optimizer=optimizer, scheduler=scheduler)
    train_losses, valid_losses = trainer.train(batch_size=2, earlystop_patience=100, n_epochs=10000)
    # testing
    test_data = [data.to(trainer.device) for data in test_data]
    x_test_pics, x_test_other, y_test = test_data
    pred = model(x_test_pics, x_test_other)
    y_pred = torch.squeeze(pred.cpu()).detach().numpy()
    test_loss = criterion(torch.squeeze(pred.cpu()), torch.squeeze(y_test.cpu())).data.item()
    r2 = r2_score(torch.squeeze(y_test.cpu()).numpy(), y_pred)
    # print(f'y_pred: {y_pred}')
    print(f'Test loss: {test_loss} | R2 score: {r2}')




