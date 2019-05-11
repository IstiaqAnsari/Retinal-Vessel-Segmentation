from torch.utils.data import TensorDataset,DataLoader,Dataset
from torch.autograd import Variable
import torch.optim as optim, matplotlib.pyplot as plt, numpy as np





def fit(Network,Epoch,trainLoader = None,Test = False,rate = 1e-3):
    net = Network
    iterations = Epoch
    trainLoss = []
    testLoss = []
    lr = rate
    start = time.time()
    criterion = nn.NLLLoss2d()
    optimizer = optim.Adam(net.parameters(),lr)
    for epoch in range(iterations):
        epochStart = time.time()
        runningLoss = 0
        net.train(True)
        for data in trainLoader:
            inputs = data[0]
            labels = data[1]
            inputs,labels = Variable(inputs),Variable(labels)
            #inputs,labels = inputs.cuda(),labels.long().cuda()
            inputs, labels = inputs.type(torch.FloatTensor).cuda(),labels.long().cuda()

            outputs = net(inputs)
            loss = criterion(F.log_softmax(outputs),labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            runningLoss += loss.item()

        avgTrainLoss = runningLoss/(trainLoader.batch_size*len(trainLoader))
        trainLoss.append(avgTrainLoss)
    
        if Test:
            net.train(False)
            test_runningLoss = 0

            for data in testLoader:
                inputs,labels = data
                inputs,labels = Variable(inputs),Variable(labels)
                inputs,labels = inputs.cuda(),labels.long().cuda()

                outputs = net(inputs)
                loss = criterion(F.log_softmax(outputs),labels)

                test_runningLoss += loss.item()

            avgTestLoss = test_runningLoss/20
            testLoss.append(avgTestLoss)

            fig1 = plt.figure(1)
            plt.plot(range(epoch+1),trainLoss,'r--',label = 'train')
            plt.plot(range(epoch+1),testLoss,'g--',label = 'test')
            if epoch == 0:
                plt.legend(loc = 'upper left')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
            epochEnd = time.time()-epochStart
            print(epoch, avgTrainLoss, avgTestLoss,epochEnd)      
        if not Test:
            epochEnd = time.time()-epochStart
            print(epoch, avgTrainLoss,epochEnd)    
    end = time.time()-start
    print(end)