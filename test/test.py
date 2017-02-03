import random,cv2, os
import numpy as np
from PIL import Image
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
recognizer = cv2.createLBPHFaceRecognizer()
timageh,timagew = 100,100#test image height and width(h,w)....
path = './yalefaces'
netinput = [timageh*timagew + 1, timagew*timageh/4 +1, (timageh+timagew)/2 + 1, timagew*timageh/4 +1, 5]
mainnet=[]
images=[]
def network():
	global mainnet
	networkb, networkw,count = [], [], 0
	for i in range(0, len(netinput)-1):
		networkw.append(np.random.rand(netinput[i],netinput[i+1]))
		networkb.append(np.random.rand(netinput[i]))
		networkb[i][0] = 1.000 # mathematical cheating......:-) 
	mainnet = [{'weights':x, 'neuron':y} for x, y in zip(networkw, networkb)]
	# print "length neuron.....",len(mainnet[0]['neuron'])
	# print "length weights.....",len(mainnet[0]['weights'][0])
	# print mainnet[0]['weights']

def get_image_data(path):
	global images
	image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.sad')]
	for image_path in image_paths:
		image_pil = Image.open(image_path).convert('L')
		image = np.array(image_pil, 'uint8')
		image = cv2.resize(image,(5*timagew, 5*timageh), interpolation = cv2.INTER_CUBIC)
		nbr = os.path.split(image_path)[1].split(".")[1]
		faces = faceCascade.detectMultiScale(image)
		for (x, y, w, h) in faces:
			if nbr == "happy":
				images.append((cv2.resize(image[y:y+h, x:x+w],(timagew, timageh), interpolation = cv2.INTER_CUBIC), 1))
			elif nbr == "sad":
				images.append((cv2.resize(image[y:y+h, x:x+w],(timagew, timageh), interpolation = cv2.INTER_CUBIC), 2))
			elif nbr == "glasses":
				images.append((cv2.resize(image[y:y+h, x:x+w],(timagew, timageh), interpolation = cv2.INTER_CUBIC), 3))
			elif nbr == "surprised":
				images.append((cv2.resize(image[y:y+h, x:x+w],(timagew, timageh), interpolation = cv2.INTER_CUBIC), 4))
			elif nbr == "sleepy":
				images.append((cv2.resize(image[y:y+h, x:x+w],(timagew, timageh), interpolation = cv2.INTER_CUBIC), 5))
			elif nbr == "wink":
				images.append((cv2.resize(image[y:y+h, x:x+w],(timagew, timageh), interpolation = cv2.INTER_CUBIC), 6))
			else:
				images.append((cv2.resize(image[y:y+h, x:x+w],(timagew, timageh), interpolation = cv2.INTER_CUBIC), 7))
			cv2.imshow("Adding faces to traning set...", cv2.resize(image[y:y+h, x:x+w],(5*timagew, 5*timageh), interpolation = cv2.INTER_CUBIC))
			cv2.waitKey(5)
		# print image
	# print images
	pass

def inputlayer_init():
	global mainnet
	i=1
	for image in images:
		count = 1
		for j in range(0, timageh):
			for k in range(0, timagew):
				mainnet[0]['neuron'][count] = image[0][j][k]
				count = count +1
				
		# print "after init input layer.................................",mainnet
		print "forwardfeed for image .................",i
		forwardfeed()
		i = i+1
	pass

def forwardfeed():
	global mainnet
	# print len(mainnet)		
	layer = 0
	while(layer < 5):
		for j in range(0, netinput[layer]):
			count = 1
			for i in range(0, len(netinput)-2):
				mainnet[i+1]['neuron'][count] = sigmoid(sum([(x*y[count-1]) for x, y in zip(mainnet[i]['neuron'], mainnet[i]['weights'])]))
				count = count + 1
		layer = layer + 1
	print mainnet
	print "##############################################################"
	# print "after feedforward second layer.................................",mainnet

def backprapogation():
	
	pass

##misc.. functiions
sigmoid = lambda z: 1.0/(1.0+np.exp(-z))
sigmoid_prime = lambda z: sigmoid(z)*(1-sigmoid(z))

print "getting training objects.................."
get_image_data(path)
print "genetating neural network with random activation_values and weights................."
network()
print "input layer init.........."
inputlayer_init()

