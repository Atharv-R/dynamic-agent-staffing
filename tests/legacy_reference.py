# ruff: noqa
"""``run_strategy`` copied verbatim from the team's Main.ipynb (MIT, (c) 2025
Dynamic-Agent-Staffing-Plan-Google-Ads), used only to check that the package's legacy mode
reproduces the original simulator."""

import math

import numpy as np
import pandas as pd


def run_strategy(threshold, current_demand, forecasted_demand, free_agents, salary,
                 alpha=1, beta=1, H=1, F=1, strategy_type = "baseline", start_date = "2023-01-01"):

    agents_pool = np.array([[0 for i in range(10)] for _ in range(free_agents)])
    adv_id = 1
    advertiser_pool = []

    start_index = current_demand.index.get_loc(start_date)
    current_date = current_demand.index[start_index]

    Costs = 0
    Rev = 0

    uplift = 0.135

    agent = [free_agents]
    agents_idle = [free_agents]

    for day in range(365):

        # Add new sign-ups to the pool
        todays_demand = current_demand.loc[current_date]["Demand"]
        todays_budget = current_demand.loc[current_date]["Budget"]
        for _ in range(int(todays_demand)):
            if math.isnan(todays_budget):
                advertiser_pool.append([0, 0, adv_id])
            else:
                advertiser_pool.append([todays_budget, 0, adv_id])
            adv_id += 1

        advertiser_pool.sort(key = lambda x: x[0], reverse = True)

        if strategy_type == "baseline":
            # Demand info about next 60 days
            demand_next60 = forecasted_demand.loc[current_date:current_date + pd.Timedelta(days = 60)]["Demand"].mean()

            # If mean budget over next 60 days is higher than threshold, hire new agents
            if demand_next60 > threshold:
                # Hire H agents
                for _ in range(H):
                    agents_pool = np.append(agents_pool, np.repeat(30,10)[np.newaxis, :], axis=0)

            else:
                # Try to fire F agents
                count_fired = 0
                idx_fired = []
                for i in range(agents_pool.shape[0]):
                    if np.sum(agents_pool[i] > 30) == 0:
                        Costs += 30 * salary/365
                        Costs += 0.4 * salary
                        idx_fired.append(i)
                        count_fired += 1
                        if count_fired == F:
                            break
                agents_pool = np.delete(agents_pool, idx_fired, axis=0)

        else:
            # Demand info about next 30 days
            demand_next30 = forecasted_demand.loc[current_date:current_date + pd.Timedelta(days = 30)]["Demand"].sum()

            slot_availability = np.argwhere(agents_pool < 30).shape[0]

            # If mean budget over next 60 days is higher than threshold, hire new agents
            if demand_next30 > beta * slot_availability:
                # Hire H agents
                hire = int((demand_next30-beta * slot_availability)/10)
                for _ in range(hire):
                    agents_pool = np.append(agents_pool, np.repeat(30,10)[np.newaxis, :], axis=0)

            elif slot_availability > alpha*demand_next30:

                # Try to fire F agents
                fire = int(max(slot_availability - alpha*demand_next30, 0)/10)
                count_fired = 0
                idx_fired = []
                for i in range(agents_pool.shape[0]):
                    if np.sum(agents_pool[i] > 30) == 0:
                        Costs += 30 * salary/365
                        Costs += 0.4 * salary
                        idx_fired.append(i)
                        count_fired += 1
                        if count_fired == fire:
                            break
                agents_pool = np.delete(agents_pool, idx_fired, axis=0)


        # Update costs
        Costs += agents_pool.shape[0] * salary/365

        # Assignment
        ids_remove = []
        zero_indices = np.argwhere(agents_pool == 0)
        curr_pos = 0
        if zero_indices.shape[0] > 0:
            for i in range(len(advertiser_pool)):
                x, y = zero_indices[curr_pos]
                curr_pos += 1
                agents_pool[x][y] = 60

                Rev += advertiser_pool[i][0] * uplift
                ids_remove.append(advertiser_pool[i][2])
                if curr_pos >= zero_indices.shape[0]:
                    break


        # Update agents pool
        for j in range(agents_pool.shape[0]):
            for spot in range(10):
                if agents_pool[j][spot] > 0:
                    agents_pool[j][spot] -= 1

        advertiser_pool = [x for x in advertiser_pool if x[2] not in ids_remove]

        # Update advertisers count
        for i in range(len(advertiser_pool)):
            advertiser_pool[i][1] += 1

        advertiser_pool = [x for x in advertiser_pool if x[1] < 60]

        current_date += pd.Timedelta(days = 1)

        agent.append(agents_pool.shape[0])

        count_idle = 0
        for agent_row in agents_pool:
            if np.sum(agent_row) == 0:
                count_idle += 1

        agents_idle.append(count_idle)

    return Rev-Costs, np.array(agent), np.array(agents_idle)
