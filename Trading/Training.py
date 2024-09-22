import pandas as pd
from TradingEnvironment import StockTradingEnv
from Trading_agent import Agent


data=pd.read_csv('MSFT.csv')
data=data[-1000:]
episodes=100
batch_size=32
env=StockTradingEnv(data)
agent=Agent(state_size=[6],strategy="double-dqn",epsilon_decay=0.9995)
reward_history=[]
for episode in range(episodes):
    state=env.reset()
    total_reward=0
    while True:
        action=agent.act(state)
        next_state,reward,done,info=env.step(action)
        agent.remember(state,action,reward,next_state,done)
        
        state=next_state
        total_reward+=reward
        
        if done:
            break
        
        if len(agent.memory)>batch_size:
            agent.train_exp_relay(batch_size)
    reward_history.append(total_reward)
    if episode>100:
        avg_reward=sum(reward_history[-100:])/100
    else:
        avg_reward=sum(reward_history[-100:])/(episode+1)
    print(f'Episode: {episode+1}, Total Reward: {total_reward},avg_reward: {avg_reward}, Epsilon:{agent.epsilon},Loss:{agent.loss_val},Money earned:{env.balance+env.holdings*env.current_price}')
        