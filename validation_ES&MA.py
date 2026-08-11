# -*- coding: utf-8 -*-
"""
Created on Mon May 12 20:26:47 2025

@author: ASUS
"""
import statistics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Parameters for the noise
mean = 0
std_dev = 10
size = 4000
np.random.seed(42)
# Generate noise from a normal distribution
samples = np.random.normal(loc=mean, scale=std_dev, size=size)
# Parameters for AR1 process
phi = 0.6
mue = 100
demand = list(range(size))

# Generate demand
demand[0] = mue + samples[0]

for i in range(1,size):
    demand[i] = (1-phi)*mue + phi*demand[i-1] + samples[i]
    

review_period = 1
K_safety = 0     
       
# defining starting state
def starting_state():
    state = {"I":0}
    data = {"backorder": list(), "Number_of_Orders":0 , "Orders": list(), 'inventory level': list()}
    future_event_list = list()
    future_event_list.append({'Event Type': 'Next Period', 'Event Time': 1}) # This is an Event Notice
    return future_event_list, state, data

# defining order up tp level
def out_lv(t, demand, lead_time):
    t_int = max(1, int(round(t, 0)))
    out_lv = (lead_time + review_period + K_safety) * demand[t_int - 1]
    #print(out_lv)
    return out_lv


def start_of_period(t, demand, state, data, future_event_list, sorted_fel, lead_time):
    # starting inventory
    if t == 1:
        state['I'] = 0
        #print("state", state)
    else:
        data['inventory level'].append(state['I'])
    # if an order is about to arrive in this period, add it to inventory
    if sorted_fel[1]['Event Type'] == "Order Arrival":
        state['I'] += sorted_fel[1]['Order Quantity']
        #print("state order", state)
        future_event_list.remove(sorted_fel[1])
    # subtract demand from inventory
    state['I'] = state['I'] - actual_demand[round(t,0)-1]
    # if inventory is negative, calculate backorder
    if state['I'] < 0:
        data["backorder"].append(-state["I"])
    else:
        data["backorder"].append(0)
        
    #print("state", state)
    future_event_list.append({'Event Type': 'Next Period', 'Event Time': int(round(t,0))+1})


def review_time(state, data, demand ,t, lead_time, future_event_list, sorted_fel, warm_up):

    out_level = out_lv(t, demand, lead_time)
    #print('out_level', out_level)
    
    # if an order is about to arrive in the coming period, consider it for inventory position
    if len(sorted_fel) > 2:
        if sorted_fel[2]['Event Type'] == "Order Arrival":
            if round(sorted_fel[2]['Event Time'],0) == round(t,0):
                current_state = state['I'] + sorted_fel[2]['Order Quantity']
            else:
                current_state = state['I']
    else:
        current_state = state['I']
    
    # calculate in transit orders that aren't for the coming period
    in_transit = 0
    if len(sorted_fel) > 2:
        for i in range(1, len(sorted_fel)):         
            if sorted_fel[i]['Event Type'] == "Order Arrival":
                if round(sorted_fel[i]['Event Time'],0) != round(t,0):
                    in_transit += sorted_fel[i]['Order Quantity']
    
    inventory_position = current_state + in_transit 
    order = max(0 , out_level-inventory_position) 
    
    if order > 0:
        future_event_list.append({'Event Type': 'Order Arrival', 'Event Time': int(round(t,0)) + lead_time + 0.00001, 'Order Quantity': order})

        if t > warm_up:
            data['Orders'].append(order)
            data['Number_of_Orders'] += 1
    
        
    future_event_list.append({'Event Type': 'Review Period', 'Event Time': int(round(t,0)) + review_period - 0.00001})

#def order_arrival(state, order):
#    state['I'] += order



def simulation(simulation_time, warm_up, lead_time, demand):
    t = 1
    future_event_list, state, data = starting_state()
    future_event_list.append({'Event Type': 'Review Period', 'Event Time': 1 - 0.00001})
    while t < simulation_time:
        sorted_fel = sorted(future_event_list, key=lambda x: x['Event Time'])
        current_event = sorted_fel[0]
        #FEL.append(sorted_fel)
        t = current_event['Event Time']
        if current_event['Event Type'] == 'Next Period':
            start_of_period(t, demand, state, data, future_event_list, sorted_fel, lead_time)
        elif current_event['Event Type'] == 'Review Period':
            review_time(state, data, demand, t, lead_time, future_event_list,sorted_fel, warm_up)


        future_event_list.remove(current_event)
    
    
    average_backorder = statistics.mean(data['backorder'][warm_up:])
    order_variance = statistics.variance(data['Orders'])
    demand_variance = statistics.variance(actual_demand[warm_up:int(round(t,0))])
    hold_inventory = [0 if i < 0 else i for i in data['inventory level']]
    average_inventory = statistics.mean(hold_inventory[warm_up:])
    hold_cost =  [i * h for i in hold_inventory]
    backorder_cost = [i * b for i in data['backorder']]
    order_cost = data["Number_of_Orders"] * k
    total_cost = sum(hold_cost[warm_up:]) + sum(backorder_cost[warm_up:]) + order_cost
    result = {'average inventory': average_inventory, 
              'average backorder': average_backorder,
              'amplification ratio': order_variance/demand_variance,
              'cost' : total_cost}
    return result, data

#validation
h = 10
b = 5
k = 0.25


df = pd.DataFrame(demand, columns=['demand'])

""" #predicting with ES
alpha_value = 0.3
model = ExponentialSmoothing(df['demand'], trend=None, seasonal=None,
                             initialization_method="known", 
                             initial_level=df['demand'].iloc[0], 
                             seasonal_periods=None).fit(
                                 smoothing_level=alpha_value, optimized=False)
                                 
model.params
# Make one-step-ahead forecasts
df['forecast'] = model.fittedvalues.shift(0)
"""
# preidicting with MA
def moving_average(data, N):
    return data.rolling(window=N).mean()

# one_step_ahead forecasting
n = 5

forecast = moving_average(df['demand'], n)
lagged_series = forecast.shift(1)  # Shift the series by 1 to create the lag

df['forecast'] = lagged_series

actual_selected = df.loc[1:,['demand']]
actual_demand = actual_selected['demand'].tolist()

selected = df.loc[1:,['forecast']]
demand = selected['forecast'].tolist()
            
result, data = simulation(3900, 200, 2, demand)
print(result)