import os, numpy as np, random, matplotlib.pyplot as plt,cv2,io,csv,pickle
from numpy import amax
from PIL import Image





path = 'data/train/'
trainPath = path
testPath = path.replace('train','test')

def exampleData():
	sampleImge = np.array(Image.open(path+'images/21_training.tif'))
	sampleGT = np.array(Image.open(path+'vein/21_manual1.gif'))
	sampleMask = np.array(Image.open(path+'mask/21_training_mask.gif'))
	plt.figure(figsize=(15,3))
	plt.subplot(131)
	plt.imshow(sampleImge)
	plt.subplot(132)
	plt.imshow(sampleGT,cmap = 'gray')
	plt.subplot(133)
	plt.imshow(sampleMask,cmap = 'gray')

def getData():
	
	
	trainList = os.listdir(path+'images')
	testList = os.listdir(testPath +'images')
	
	
def createTrainData():
	TrainImages = torch.FloatTensor(600,3,224,224)
	TrainLabels = torch.FloatTensor(600,224,224)
	trainList = os.listdir(path+'images')

	img_no = 0
	for file in trainList:
		imgNum = file.split('_')[0]
		im = Image.open(trainPath+'images/'+file)
		seg = Image.open(trainPath+'vein/'+str(imgNum)+'_manual1.gif')
		mask = Image.open(trainPath+'mask/'+str(imgNum)+'_training_mask.gif')
		
	#     im = np.array(im.resize((224,224)))/255
	#     seg = np.array(seg.resize((224,224)))/255
	#     mask = (np.array(mask.resize((224,224)))-seg)/255
	#     idx = np.where(mask == 1)t
	#     seg[idx] = 2
		
	#     TrainImages[img_no] = torch.from_numpy(im).ranspose(0,2).unsqueeze(0)
	#     TrainLabels[img_no] = torch.from_numpy(seg).transpose(0,1).unsqueeze(0)
	#     img_no += 1
		
		
		im = np.array(im)/255
		seg = np.array(seg)/255
		mask = (np.array(mask)-seg)/255
		idx = np.where(mask == 1)
		seg[idx] = 2
		
		randIdx1 = np.random.randint(0,im.shape[0]-224,30)
		randIdx2 = np.random.randint(0,im.shape[1]-224,30)
		
		for p in range(30):
			patch = im[randIdx1[p]:randIdx1[p]+224,randIdx2[p]:randIdx2[p]+224,:]
			seg_patch = seg[randIdx1[p]:randIdx1[p]+224,randIdx2[p]:randIdx2[p]+224] 
			TrainImages[img_no] = torch.from_numpy(patch).transpose(0,2)
			TrainLabels[img_no] = torch.from_numpy(seg_patch).transpose(0,1)
			img_no += 1
	
def createTestData():
	testList = os.listdir(testPath +'images')
	TestImages = torch.FloatTensor(20,3,224,224)
	TestLabels = torch.FloatTensor(20,224,224)
	img_no = 0
	for file in testList:
		imgNum = file.split('_')[0]
		im = Image.open(testPath+'images/'+file)
		seg = Image.open(testPath+'1st_manual/'+str(imgNum)+'_manual1.gif')
		mask = Image.open(testPath+'mask/'+str(imgNum)+'_test_mask.gif')
		
		im = np.array(im.resize((224,224)))/255
		seg = np.array(seg.resize((224,224)))/255
		mask = (np.array(mask.resize((224,224)))-seg)/255
		idx = np.where(mask == 1)
		seg[idx] = 2
		
		TestImages[img_no] = torch.from_numpy(im).transpose(0,2).unsqueeze(0)
		TestLabels[img_no] = torch.from_numpy(seg).transpose(0,1).unsqueeze(0)
		img_no += 1		
