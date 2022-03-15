


# X = <X1, X2, 1 (bias)>
x = np.array([[1,1,1],[1,0,1],[0,1,1],[0,0,1]])
y = np.array([[0,1,1,0]]).T

class Perceptron:
    def __init__(self, train, hidden, outputs, target, lr=0.01, iters=10000):
        self.num_inputs = train.shape[1] 
        self.hidden = hidden
        self.output = outputs
        self.x = train
        self.y = target
        self.iters = iters
        self.lr = lr
        
        self.w1 = np.random.uniform(size=(self.num_inputs, self.hidden))
        self.w2 = np.random.uniform(size=(self.hidden, self.output))

    def sigmoid(self,x):
        """
        Sigmoid activation:
        """

        return 1/(1+np.exp(-x)) 

    def classify(self, x):
        """
        Forward pass. 
        """
        z0 = np.dot(x, self.w1)         
        a0 = self.sigmoid(z0)               
        z1 = np.dot(a0,self.w2)                      

        return self.sigmoid(z1) 

    def train(self):
        for __ in range(self.iters):
            z0 = np.dot(self.x, self.w1)    #(Nx4)
            a0 = self.sigmoid(z0)           #(Nx4)
            z1 = np.dot(a0,self.w2)         #(Nx1)
            a1 = self.sigmoid(z1)           #(Nx1)

            delta_w2 = (a1-self.y) * a1*(1-a1)              #(Nx1)
            delta_w1 = np.dot(delta_w2, self.w2.T) * (1-a0) #(Nx4)

            self.w1 -= self.lr*np.dot(self.x.T,delta_w1)    #(3x4)
            self.w2 -= self.lr*np.dot(a0.T,delta_w2)        #(4x1)


nn = Perceptron(train=x, hidden=4, outputs=1, target=y, lr=0.1, iters=30000)
nn.train()
nn.classify(x)


