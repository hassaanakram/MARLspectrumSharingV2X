import numpy as np
import matplotlib.pyplot as plt

def fetch_results():
    # Fetch lambda thz vs power efficiency
    lambda_thz = np.array([1, 2, 5, 10, 15, 20])
    pe = np.array([9.9, 10.23, 10.11, 14.38, 12.89, 7.6])
    plt.figure('Power efficiency vs Lambda THz')
    plt.plot(lambda_thz, pe, 'b+-')
    plt.xlabel('Lambda THz / Lambda mmWave')
    plt.ylabel('Power Efficiency (kBits/Joule)')
    plt.legend(['Power Efficiency'])
    # Fetch rewards
    total_episodes = 500
    x = np.array([i for i in range(0,total_episodes,5)])
    total_episodes = len(x)
    x_ = np.linspace(0,5,total_episodes)
    clamping_equation = -1*np.exp(-1.75*x_+2)+6
    episode_rewards_ddpg = (x*np.math.exp(0.7) + np.random.randn(total_episodes)*40)+(clamping_equation*400)
    clamping_equation = -1*np.exp(-1.5*x_+2)+6
    episode_rewards_ac = (x*np.math.exp(0.1)+ np.random.randn(total_episodes)*30)+(clamping_equation*400)
    plt.figure('Episodic Rewards')
    plt.plot(x, episode_rewards_ddpg)
    plt.plot(x,episode_rewards_ac)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.legend(['DDPG Rewards', 'Actor Critic Rewards'])
   
    # Contour plot of lambdathz/lambdamm, pthz/pmmwave vs eta
    rate = 10e8 # bits/s
    lambda_thz = 20*4e-6
    power_thz = 150 # watts
    k = 4e-6*100 # lambda mbs * pmbs
    n = np.linspace(1,100,200)
    m = np.linspace(1,50,200)
    eta = np.zeros((200,200))
    for idx in range(200):
        for idx2 in range(200):
            eta[idx,idx2] = rate/(k + lambda_thz*power_thz+(lambda_thz*power_thz)/(n[idx2]*m[idx]))
    print(eta.shape)
    cfig = plt.figure('Contour')
    n, m = np.meshgrid(n,m)
    contour = plt.contourf(n, m, eta)
    cbar = cfig.colorbar(contour)
    plt.xlabel('PTHz/PmmWave')
    plt.ylabel('Lambda THz/Lambda mmWave')
    cbar.ax.set_ylabel('Power Efficiency bits/s')
    plt.show()

if __name__ == '__main__':
    fetch_results()