import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd   # pandas is not used in this notebook
import cantera.data

gas1 = ct.Solution('gri30.yaml')  # note the use of the ct. prefix to access the Cantera object "Solution"

# Get all of the Species objects defined in the GRI 3.0 mechanism
species = {S.name: S for S in ct.Species.list_from_file("gri30.yaml")}

# Create an IdealGas object including incomplete combustion species
gas1 = ct.Solution(thermo="IdealGas", species=species.values())


#Constant pressure combustion  at 1 atm and 298K
phi = np.arange(0.5, 1.75, .25)
T_flame_cp = np.zeros(phi.shape)
for i in range(len(phi)):
    gas1.TP = 298, ct.one_atm
    gas1.set_equivalence_ratio(phi[i], "CH4", "O2:2, N2:7.52") #Changed it from O2:1 to 2 because of
    gas1.equilibrate("HP")
    T_flame_cp[i] = gas1.T
    print("Constant pressure combustion at 1atm and 298 K. Equivalence Ratio is:" + str(phi[i]))
    print(gas1[i]())
print(T_flame_cp)

## For phi = 1atm, 298K: phi=.5, you have O2, H2O, CO2 and N2 in quantities with molar fraction above .1% . 
# This has the same major products in the major-minor species model.
# For phi =.75,  you have N2, NO, CO2, H2O, OH, O2. This does not have all of the major products but has minor species.
#For phi = 1, you have H2, O2, OH, H2O, CO, CO2, NO, N2. This follows the major-species model with some minor species.
#For phi = 1.25, you have H2, H2O,CO, CO2, N2. Has most of the major products but O2.
#For phi = 1.5, you have H2, H2O, CO, CO2, N2

#____________________________________________________________________________________________________________________________
#______________________________Constant Volume Combustion at 1 atm and 298K_________________________________________________#
T_flame_cv = np.zeros(phi.shape)
for i in range(len(phi)):
    gas1.TP = 298, ct.one_atm
    gas1.set_equivalence_ratio(phi[i], "CH4", "O2:2, N2:7.52") #Changed it from O2:1 to 2 because of
    gas1.equilibrate("UV")
    T_flame_cv[i] = gas1.T
    print(" Constant Volume combustion at 1 atm and 298 K. Equivalence Ratio is:" + str(phi[i]))
    print(gas1[i]())
print(T_flame_cv)


# Plots for 2ai, 2aii
plt.plot(phi, T_flame_cp, label="Constant Pressure Combustion")
plt.plot(phi, T_flame_cv, label="Constant Volume Combustion", lw=2)

plt.title("Flame Temperature vs Equivalence Ratio at Constant Pressure")
plt.grid(True)
plt.xlabel("Equivalence ratio, $\phi$")
plt.ylabel("Temperature [K]")
plt.legend(['Constant Pressure Combustion','Constant Volume Combustion'], loc='upper left')

plt.show()

## For phi = 1atm, 298K: phi=.5, you have O2, H2O, CO2, NO and N2 in quantities with molar fraction above .1% . 
# This has the same major products in the major-minor species model.
# For phi =.75,  you have O2, OH, H2O, CO, CO2, NO, N2. This does not have all of the major products but has minor species.
#For phi = 1, you have H2, O2, OH, H2O, CO, CO2, NO, N2. This follows the major-species model with some minor species.
#For phi = 1.25, you have H2, H, OH, H2O,CO, CO2, N2. Has most of the major products but O2.
#For phi = 1.5, you have H2, H, H2O, CO, CO2, which has most of the major products in the model.

#___________________________constant pressure combustion at 1 atm and initial temperature of 500 K_____________________________________ 
T_flame_cp2 = np.zeros(phi.shape)

for i in range(len(phi)):
    gas1.TP = 500, ct.one_atm
    gas1.set_equivalence_ratio(phi[i], "CH4", "O2:2, N2:7.52") #Changed it from O2:1 to 2 because of
    gas1.equilibrate("HP")
    T_flame_cp2[i] = gas1.T
    print(" Constant Pressure combustion at 1 atm and initial temperature of 500K. Equivalence Ratio is:" + str(phi[i]))
    print(gas1[i]())
print(T_flame_cp2)



## For phi = 1atm, 500K: phi=.5, you have O2, H2O, CO2, NO and N2 in quantities with molar fraction above .1% . 
# This has the same major products in the major-minor species model.
# For phi =.75,  you have O2, OH, H2O,  CO2, NO, N2. This does not have all of the major products but has minor species.
#For phi = 1, you have H2, O2, OH, H2O, CO, CO2, NO, N2. This follows the major-species model with some minor species.
#For phi = 1.25, you have H2, H, H2O,CO, CO2, N2. Has most of the major products but O2.
#For phi = 1.5, you have H2,  H2O, CO, CO2, N2 which has most of the major products in the model.



#______________________constant pressure combustion at 10 atm and initial temperature of 298 K _________________
T_flame_cp3 = np.zeros(phi.shape)

for i in range(len(phi)):
    gas1.TP = 298, 10 * ct.one_atm
    gas1.set_equivalence_ratio(phi[i], "CH4", "O2:2, N2:7.52") #Changed it from O2:1 to 2 because of
    gas1.equilibrate("HP")
    T_flame_cp3[i] = gas1.T
    print("Constant Pressure combustion at 10 atm and initial temperature of 298 . Equivalence Ratio is:" + str(phi[i]))
    print(gas1[i]())
print(T_flame_cp3)


# Plots for 2aiii, 2aiv
plt.plot(phi, T_flame_cp2, label="1 atm, 500K")
plt.plot(phi, T_flame_cp3, label="10 atm, 298K")

plt.title("Flame Temperature vs Equivalence Ratio")
plt.grid(True)
plt.xlabel("Equivalence ratio, $\phi$")
plt.ylabel("Temperature [K]")
plt.legend(['1 atm, 500K Combustion','10 atm, 298K Combustion'], loc='upper left')

plt.show()

## For phi = 10 atm, 298K: phi=.5, you have O2, H2O, CO2, and N2 in quantities with molar fraction above .1% . 
# This has the same major products in the major-minor species model.
# For phi =.75,  you have O2, H2O,  CO2, NO, N2. This does not have all of the major products but has minor species.
#For phi = 1, you have H2, O2, OH, H2O, CO, CO2, NO, N2 This follows the major-species model with some minor species.
#For phi = 1.25, you have H2, H2O,CO, CO2, N2. Has most of the major products but O2.
#For phi = 1.5, you have H2,  H2O, CO, CO2, N2 which has most of the major products in the model.
#__________________________________________________________________________________________________________
#b. Explain the reasons for the trends of flame temperature with: 
#i. equivalence ratio 
### Answer: As equivalence ratio increases, we have more moles of fuel to be combusted. As a result, there is less dissassociation.
## With dissassociation, there is a reduction in heat release across the whole reaction. We have the opposite here: with less dissassociation,
## more energy is released, which contributes to the production of more products and a higher flame temperature


#ii. constant pressure vs. constant volume combustion 
### Answer: The flame temperature is higher at constant volume because the energy that typically goes into work 
### ( to change volume) is no longer there and thus goes into heating up the mixture, following the energy conservation law.
###In other words, with a changing volume, some energy goes into work to make that happen. 
###If there is no work needed (due to constant volume), it heats up the mixture, resulting in higher flame temperatures.


#iii. lower vs. higher initial temperature 
###Answer: With a higher initial temperature, the reaction favors dissassociation which reduces heat release across the overall reaction.
### With a higher initial temperature, we also have a higher flame temperature because of heat released, causing the temperature to go up. 
###However, due to dissassociation, the temperature delta isn't as high because there is a reduction in heat released across the reaction.

#iv. lower vs. higher initial pressure 
###Answer: With lower pressures, dissassociation is favored. As a result, with lower pressures, there is a reduction in heat release
### across the reaction.