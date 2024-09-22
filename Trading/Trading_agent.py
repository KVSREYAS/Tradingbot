import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import numpy as np
import copy

class DQN_network(nn.Module):
    def __init__(self,lr,input_dims,output_dims):
        super(DQN_network,self).__init__()
        self.fc_1=nn.Linear(*input_dims,128)
        self.fc_2=nn.Linear(128,256)
        self.fc_3=nn.Linear(256,256)
        self.fc_4=nn.Linear(256,128)
        self.fc_5=nn.Linear(128,output_dims)
        
        self.lr=lr
        self.loss=nn.HuberLoss()
        self.optimizer=optim.Adam(self.parameters(),lr=lr)
        
        self.device=T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self,state):
        x=F.relu(self.fc_1(state))
        x=F.relu(self.fc_2(x))
        x=F.relu(self.fc_3(x))
        x=F.relu(self.fc_4(x))
        actions=self.fc_5(x)
        
        return actions
    
    
class Agent:
    def __init__(self,state_size,strategy="t-dqn",reset_every=1000,pretrained=False,model_name=None,epsilon_decay=0.995):
        self.strategy=strategy
        
        self.state_size=state_size
        self.action_size=3
        self.model_name=model_name
        self.inventory=[]
        self.memory=deque(maxlen=10000)
        self.first_iter=True
        
        
        
        self.model_name=model_name
        self.gamma=0.95
        self.epsilon=1
        self.epsilon_decay=epsilon_decay
        self.epsilon_min=0.01
        self.learning_rate=0.001
        self.loss=nn.HuberLoss()
        self.custom_objects={"huber_loss":nn.HuberLoss()}
        
        self.loss_val=0
        
        if pretrained and self.model_name is not None:
            self.model=self.load()
        else:
            self.model=DQN_network(lr=self.learning_rate,input_dims=self.state_size,output_dims=self.action_size)
            
        if self.strategy in ["t-dqn","double-dqn"]:
            self.n_iter=1
            self.reset_every=reset_every
            
            self.target_model=copy.deepcopy(self.model)
    
    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))
        
    def act(self,state,is_eval=False):
        if not is_eval and random.random() <=self.epsilon:
            return random.randrange(self.action_size)
        
        if self.first_iter:
            self.first_iter=False
            return 1
        device=T.device("cuda:0" if T.cuda.is_available() else "cpu")
        state_tensor=T.FloatTensor(state).to(device)
        action_probs=self.model.forward(state_tensor)
        return T.argmax(action_probs).item()
    
    def train_exp_relay(self,batch_size):
        batch=random.sample(self.memory,batch_size)
        X_train,y_train = [],[]
        states,actions,rewards,next_states,dones=zip(*batch)
        
        batch_index=np.arange(batch_size,dtype=np.int32)
        device=T.device("cuda:0" if T.cuda.is_available() else "cpu")
        states = T.FloatTensor(states).to(device)
        next_states = T.FloatTensor(next_states).to(device)
        rewards = T.FloatTensor(rewards).to(device)
        dones = T.FloatTensor(dones).to(device)
        
        if self.strategy=="dqn":
            with T.no_grad():
                max_next_q_values=self.model.forward(next_states).max(dim=1)[0]
            max_next_q_values[dones==1]=0.0
            target_values=rewards+self.gamma*max_next_q_values
            
            
        elif self.strategy=="t-dqn":
            if self.n_iter%self.reset_every==0:
                self.target_model.load_state_dict(self.model.state_dict())
            with T.no_grad():
                max_next_q_values=self.target_model.forward(next_states).max(dim=1)[0]
            max_next_q_values[dones==1]=0.0
            target_values=rewards+self.gamma*max_next_q_values
            
        elif self.strategy=="double-dqn":
            if self.n_iter%self.reset_every==0:
                self.target_model.load_state_dict(self.model.state_dict())
            with T.no_grad():
                action=T.argmax(self.model.forward(next_states),dim=1)
                max_next_q_values=self.target_model.forward(next_states)[batch_index,action]
            max_next_q_values[dones==1]=0.0
            target_values=rewards+self.gamma*max_next_q_values
            
        else:
            raise NotImplementedError()
        

        predicted_vals=self.model.forward(states)[batch_index,actions]
        
        self.model.optimizer.zero_grad()
    
        loss=self.loss(predicted_vals,target_values)
        self.loss_val=loss
        loss.backward()
        self.model.optimizer.step()
        
        if self.epsilon>self.epsilon_min:
            self.epsilon*=self.epsilon_decay
            
        return loss

    # def save(self,episode):
    #     str1="models/episode"+str(episode)
    #     T.save(self.model,str1)
    # def load(self):
    #     return T.load("models/episode100")
                
            
        
            
            
                
                
            
                
        

