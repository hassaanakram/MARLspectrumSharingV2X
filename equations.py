from scipy.stats import nakagami
import matplotlib.pyplot as plt
import numpy as np
import math

fig, ax = plt.subplots(1, 1)

speed_of_light = 299792458

####################################################################################################

#Nakagami, Have to check if normalized or not
nu = 4.97; #uq
#Probability Density Function
nakagami_pdf_x = np.linspace(nakagami.ppf(0.01, nu),
                nakagami.ppf(0.99, nu), 100) #ppf is percent point function for percentiles
ax.plot(nakagami_pdf_x, nakagami.pdf(nakagami_pdf_x, nu),
       'r-', lw=5, alpha=0.6, label='nakagami pdf')

#Random Numbers
random_nakagami_numbers = nakagami.rvs(nu, size=1000)

####################################################################################################

#Path Loss Model for sub-6GHz band    
lambda_MBS = 10000000
path_loss_exponent = 3
shadow_fading = 1
distance_MBS = 1000

path_loss_MBS = 20 * math.log((4*math.pi)/lambda_MBS) + (10*path_loss_exponent*math.log(distance_MBS)) + shadow_fading

#Path Loss Model for mmWave

carrier_frequency_mmWave = 100000
path_loss_exponent_NLOS = 3
path_loss_exponent_LOS = 3
distance_mmWave = 1000
fixed_path_loss = 32.4 + (20*math.log(carrier_frequency_mmWave))
shadow_fading_NLOS = 1
shadow_fading_LOS = 1

#LOS 
path_loss_mmWave_LOS = fixed_path_loss + (10*path_loss_exponent_LOS*math.log(distance_mmWave)) + shadow_fading_LOS

#NLOS
path_loss_mmWave_NLOS = fixed_path_loss + (10*path_loss_exponent_NLOS*math.log(distance_mmWave)) + shadow_fading_NLOS


#Path Loss Model for THz

carrier_frequency_THz = 10000000
distance_THz = 1000
molecular_absorption_coefficient = 3 #k_f

path_loss_spread = 20*math.log((4*math.pi*carrier_frequency_THz*distance_THz)/speed_of_light)
path_loss_absorption = math.e**(molecular_absorption_coefficient*distance_THz)
path_loss_THz = path_loss_spread + path_loss_absorption

####################################################################################################

#Received powers 
#To be added

####################################################################################################

#Performance Analysis
recieved_powers = [] #to be added
recieved_power_user = np.argmax(recieved_powers)
total_users = 1000
total_users_tier = [] #to be added
cumulative_tier_associativities = []

for users_tier in total_users_tier:
    cumulative_associativity = total_users / users_tier
    cumulative_tier_associativities.append(cumulative_associativity)


    


