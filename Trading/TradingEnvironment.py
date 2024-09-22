class StockTradingEnv:
    def __init__(self,data,initial_balance=10000):
        self.data=data
        self.initial_balance=initial_balance
        self.current_step=0
        self.balance=initial_balance
        self.holdings=0
        self.net_worth=initial_balance
        self.max_steps=len(data)
    
    def reset(self):
        self.current_step=0
        self.balance=self.initial_balance
        self.holdings=0
        self.net_worth=self.initial_balance
        state=self.data.iloc[self.current_step].values[1:]
        return list(state)
    
    def get_state(self):
        state=self.data.iloc[self.current_step].values[1:]
        return list(state)
    
    def step(self,action):
        
        current_price = self.data.iloc[self.current_step]['Close']
        self.current_price=current_price
        reward=0
        if action==1:#buy
            if self.balance>current_price:
                self.previous_balance=self.balance
                self.holdings+=self.balance/current_price
                self.balance=0
                
        elif action==2:#sell
            if self.holdings>0:
                self.balance+=self.holdings*current_price
                self.holdings=0
                
                self.net_worth=self.balance+self.holdings*current_price
                reward=self.net_worth-self.initial_balance
        
        self.current_step+=1
        done=self.current_step>=self.max_steps-1
        
        return self.get_state(),reward,done,{}
        