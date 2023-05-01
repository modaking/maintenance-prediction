import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from scipy.integrate import quad
from scipy.stats import weibull_min


def data_preparation(data):
    df = data
    x = df['Time'].tolist()
    y = []

    # Get the durations values for the durations column. The last index has duration of 0
    for i in x:
        if i == x[0]:
            z = i
            y.append(z)
        else:
            n = x.index(i)
            n = n - 1
            c = x[n]
            b = x.index(i)
            b = x[b]
            z = b - c
            y.append(z)

    df['Duration'] = y
    df2 = df.sort_values('Duration', ascending=True, ignore_index=True)

    return df2


def create_kaplanmeier_data(data):
    df = data
    x = df['Time'].tolist()
    rows_number = len(x)
    initial_probability = 1 / rows_number
    probability = initial_probability
    y = []

    for j in range(0, rows_number):
        y.append(probability)

    df['Probability'] = y
    new_probs = []

    for g in df.index:
        if df['Event'][g] == 'PM':
            h = g + 1
            hg = 107 - h
            gg = df['Probability'][g] / hg
            probability = probability + gg
            new_probs.append(0)

        else:
            new_probs.append(probability)

    censored = df[df['Event'] == 'PM'].index
    df['Probability'] = new_probs
    df.drop('Probability', inplace=True, axis=1)
    df['Probability'] = new_probs

    duplicateRows = df[df.duplicated(['Duration'])]
    dup_list = []
    # Deal with duplicates
    for cg in duplicateRows.index:
        if duplicateRows['Duration'][cg] not in dup_list:
            dup_list.append(duplicateRows['Duration'][cg])

    dup_column = duplicateRows['Duration'].tolist()

    occurrence = []

    for k in dup_list:
        nbv = dup_column.count(k)
        nbv += 1
        occurrence.append(nbv)

    bb = []
    points = df['Probability'].tolist()

    for cg in df.index:
        if df['Duration'][cg] in dup_list:
            if cg not in duplicateRows.index:
                vn = points[cg]
                ele = df['Duration'][cg]
                ind = dup_list.index(ele)
                mult = occurrence[ind]
                vn = vn * mult
                bb.append(vn)
            else:
                bb.append(0)
        else:
            vn = points[cg]
            bb.append(vn)

    df.drop('Probability', inplace=True, axis=1)
    df['Probability'] = bb

    # Add reliability

    reliable = 1
    rel_list = []

    for e in bb:
        reliable -= e
        rel_list.append(reliable)

    df['Reliability'] = rel_list

    return df


def meantimebetweenfailure_KM(data):
    df = data
    prob_list = df['Probability'].tolist()
    dur_list = df['Duration'].tolist()
    mean_list = []

    for i in dur_list:
        ind = dur_list.index(i)
        value = i * prob_list[ind]
        mean_list.append(value)

    count = 0
    for i in mean_list:
        count += i

    return count


def fit_weibull_distribution(data):
    df1 = data
    # create arrays for each parameter using linspace
    lambda_vals = np.linspace(1, 35, num=35, dtype=int)
    k_vals = np.linspace(0.1, 3.5, num=35)

    # use meshgrid to create a matrix of all parameter combinations
    lambda_mesh, k_mesh = np.meshgrid(lambda_vals, k_vals)
    param_matrix = np.concatenate((lambda_mesh.reshape(-1, 1), k_mesh.reshape(-1, 1)), axis=1)

    # create a pandas DataFrame from the matrix
    df2 = pd.DataFrame(data=param_matrix, columns=['lambda', 'kappa'])

    # add duration columns
    dur_list = df1['Duration'].tolist()
    lambda_column = df2['lambda'].tolist()
    kappa_column = df2['kappa'].tolist()

    for each in dur_list:
        weib_column = []
        for l in lambda_column:
            ind = lambda_column.index(l)
            k = kappa_column[ind]

            # calculate the reliability using the formula with logarithms
            ln_R = -(each / l) ** k
            R_t = math.exp(ln_R)

            weib_column.append(ln_R)

        name = "observation_" + str(dur_list.index(each))
        df2[name] = weib_column

        # Define columns to exclude from sum
        exclude_cols = ['Lambda', 'Kappa']

        # sum the elements of each row , excluding column 'lambda' and 'kappa'
        row_sums = df2.iloc[:, ~df2.columns.isin(exclude_cols)].sum(axis=1)

        df2['Loglikelihood_sum'] = row_sums

        # get the index of the row with the highest value in column 'Log_sum'
        max_row_index = df2['Loglikelihood_sum'].idxmax()

        # get the optimal value of lambda and kappa
        lambda_value = df2.iloc[max_row_index, 0]
        kappa_value = df2.iloc[max_row_index, 1]
    return lambda_value, kappa_value


def meantimebetweenfailure_weibull(lamb, kap):
    # calculate the mean time before failure using the formula
    MTBF = lamb * math.gamma(1 + 1 / kap)

    return MTBF


def create_weibull_curve_data(data, lamb, kap):
    df = data
    # get the index of the row with the highest value in column 'Duration'
    max_row_index = df['Duration'].idxmax()
    highest_dur = df.iloc[max_row_index, 3]

    # Define the start and end duration values
    start_duration = 0
    end_duration = highest_dur

    # Define the number of points in the duration range
    num_points = 100

    # Create the duration range
    durations = np.linspace(start_duration, end_duration, num_points)

    # Calculate the reliability function for each duration value
    reliability = np.exp(-(durations / lamb) ** kap)

    # Create a Pandas DataFrame with the duration and reliability values
    df1 = pd.DataFrame({'t': durations, 'R_t': reliability})
    print(df1)

    return df1


def visualization(data1, data2):
    km_data = data1
    weib_data = data2

    km_dur = km_data['Duration'].tolist()
    km_rel = km_data['Reliability'].tolist()
    weib_dur = weib_data['t'].tolist()
    weib_rel = weib_data['R_t'].tolist()

    figure, axis = plt.subplots(1, 2)

    # Create a plot with both the Kaplan-Meier and Weibull reliability functions
    axis[0].plot(km_dur, km_rel)
    axis[0].set_title("Kaplan-Meier")
    axis[1].plot(weib_dur, weib_rel, color='red')
    axis[1].set_title("Weibull")

    plt.show()

    return None


def create_cost_data(data1, lamb, kap, pm_cost, cm_cost):
    df = data1
    # get the index of the row with the highest value in column 'Duration'
    max_row_index = df['Duration'].idxmax()
    highest_dur = df.iloc[max_row_index, 3]

    # Define the Weibull reliability function
    def weibull_reliability(ti, lam, k):
        return math.exp(-((ti / lam) ** k))

    # Define the Weibull PDF function
    def weibull_pdf(ti, lam, k):
        return (k / lam) * ((ti / lam) ** (k - 1)) * math.exp(-((ti / lam) ** k))

    def mean_cycle_length(age, lam, k):
        t1 = age * lam
        t2 = (age + 1) * lam
        F1 = weibull_reliability(t1, lam, k)
        F2 = weibull_reliability(t2, lam, k)
        f = weibull_pdf(t1, lam, k)
        return (1 / (F1 - F2)) * ((t1 * (1 - F1)) + ((f / k) * (lam - t1 * (F2 - F1))))

    # Define the optimization function
    def optimize_cost_rate(fra):

        # Get the index of the row with the minimum value in Mean_cost
        min_index = fra.iloc[1:, :]['Mean_cost'].idxmin()

        # Get the row with the minimum value in Mean_cost
        min_row = fra.loc[min_index]

        min_cost_rate = min_row['Mean_cost']
        optimal_age = min_row['Maintenance Age']
        return optimal_age, min_cost_rate

    # Define the cost of maintenance
    maintenance_cost = pm_cost

    # Define the cost of repair per failure
    repair_cost = cm_cost

    # Define the duration of each maintenance interval
    maintenance_interval = 1.0

    # Define the range of maintenance ages to consider
    start_age = 0.0
    end_age = highest_dur
    num_ages = 100

    # Create a Pandas DataFrame to store the results
    cost_data = pd.DataFrame({'Maintenance Age': np.linspace(start_age, end_age, num_ages)})

    # Calculate the failure rate and expected number of failures for each maintenance age

    xp = cost_data['Maintenance Age']
    failure = []
    reliable = []
    mean_charge = []
    mean_cycle = []
    value_rate = []
    for i in xp:
        if i == 0:
            failure_rate = weibull_pdf(i, lamb, kap)
            reliability = weibull_reliability(i, lamb, kap)
            mean_cost = repair_cost * failure_rate + maintenance_cost * reliability
            mean_length = mean_cycle_length(i, lamb, kap)
            charge = 0
            failure.append(failure_rate)
            reliable.append(reliability)
            mean_charge.append(mean_cost)
            mean_cycle.append(mean_length)
            value_rate.append(charge)

        else:
            failure_rate = weibull_pdf(i, lamb, kap)
            reliability = weibull_reliability(i, lamb, kap)
            mean_cost = repair_cost * failure_rate + maintenance_cost * reliability
            mean_length = mean_cycle_length(i, lamb, kap)
            charge = mean_cost / mean_length
            failure.append(failure_rate)
            reliable.append(reliability)
            mean_charge.append(mean_cost)
            mean_cycle.append(mean_length)
            value_rate.append(charge)

    cost_data['Reliability'] = reliable
    cost_data['Failure_rate'] = failure
    cost_data['Mean_cost'] = mean_charge
    cost_data['Mean_length'] = mean_cycle
    cost_data['Cost_rate'] = value_rate

    vx = cost_data['Mean_cost']

    vx = cost_data['Mean_length']

    optimum_age, cost_rate = optimize_cost_rate(cost_data)

    # Create the plot
    km_dur = cost_data['Maintenance Age'].tolist()
    km_rel = cost_data["Cost_rate"].tolist()

    plt.plot(km_dur, km_rel, label='Cost Rates')
    plt.xlabel("Age")
    plt.ylabel("Cost rate")
    plt.title("Cost rates by age")
    plt.show()

    return optimum_age, cost_rate


def CBM_data_preparation(data):
    df = data

    # calculate the time increments
    df['increments'] = np.diff(df['Condition'], prepend=0)

    # filter out rows with negative increments

    df_filtered = df.loc[df['increments'] >= 0]
    df_filtered = df_filtered.reset_index(drop=True)

    return df_filtered


def CBM_create_simulations(data, level, thresh):
    sim_data = data

    # Define simulation parameters
    num_simulations = 1000  # Number of simulation iterations
    time_step = 1  # Time step size
    threshold = thresh  # Threshold for condition-based maintenance
    failure_rate = level  # Failure rate per time period
    condition = 0
    time = 0

    # Set up data frame to store results
    results = pd.DataFrame(columns=["duration", "Event"])

    # Loop over simulations
    for i in range(num_simulations):
        time = 0
        condition = 0

        # Create infinite loop for each simulation
        while True:
            # Increase time by fixed value
            time += time_step

            # Increase condition by random float
            condition_increment = np.random.choice([5.05, 4.1, 8.15, 2.2, 6.3, 7.5, 2.1, 1.2, 9.3])
            condition += condition_increment

            # Check if machine has failed
            if condition >= failure_rate:
                duration = time
                event = "failure"
                results = results.append({"duration": duration, "Event": event}, ignore_index=True)

                condition = 0
                break

            # Check if preventive maintenance is needed
            if condition >= threshold:
                duration = time
                event = "PM"
                results = results.append({"duration": duration, "Event": event}, ignore_index=True)
                condition = 0
                break

    return results


def CBM_analyse_costs(data, pm, cm):
    df = data
    print(df.head(40))

    def average_cycle_duration(duration):
        return sum(duration) / len(duration)

    durations = df['duration'].tolist()

    av_cycle = average_cycle_duration(durations)
    print(av_cycle)

    # Count the number of rows where 'Event' is equal to 'failure'
    failure_count = (df['Event'] == 'failure').sum()
    failure_fraction = failure_count / len(durations)
    preventive = 1 - failure_fraction

    ff = failure_fraction * len(durations)
    pf = len(durations) - ff

    sum_cost = ff * cm + pf * pm
    av_cost = sum_cost / len(durations)
    cost_rate = av_cost / av_cycle
    print(cost_rate)

    return cost_rate


def CBM_create_cost_data(data, pm, cm, level):
    count_list = [0.75, 0.8, 0.85, 0.9, 0.95]
    thresh_list = []
    for i in count_list:
        bg = i * level
        thresh_list.append(bg)

    cost_rates = []

    for i in thresh_list:
        sim_df = CBM_create_simulations(data, level, i)
        rate = CBM_analyse_costs(sim_df, pm, cm)
        cost_rates.append(rate)

    new_df = pd.DataFrame({'Threshold': thresh_list, 'Cost_rate': cost_rates})

    # Get the index of the row with the minimum value in 'Threshold'
    min_index = new_df['Threshold'].idxmin()

    # Get the row with the minimum value in 'Threshold'
    min_row = new_df.loc[min_index]

    min_thresh = min_row['Threshold']
    min_cost = min_row['Cost_rate']

    # Create the plot
    km_dur = new_df["Threshold"].tolist()
    km_rel = new_df["Cost_rate"].tolist()

    print(km_dur)
    print(km_rel)
    # Plot the parameters
    plt.plot(km_dur, km_rel, label='Cost Rates')
    plt.xlabel("Threshold")
    plt.ylabel("Cost rate")
    plt.title("Cost rates by Threshold")
    plt.show()

    return min_cost, min_thresh


def run_analysis():
    count = 1
    machines = [1, 2, 3]
    choices = ['yes', 'no']
    # get user inputs
    while count:

        machine = input('Enter machine to analyse btwn[1,2,3]')

        try:
            x = machine in machines
            count = 0
            machine = str(machine)
            machine = machine.strip()

        except x is False:
            print("Invalid input.")

    count = 1
    while count:

        pom = input('Enter preventive maintenance cost for this machine?')

        try:
            pom = float(pom)
            count = 0

        except ValueError:
            print("Invalid input.")

    count = 1
    while count:

        com = input('Enter corrective maintenance cost for this machine?')

        try:
            com = float(com)
            count = 0

        except ValueError:
            print("Invalid input.")

    count = 1
    while count:

        cond_opt = input('Do we want to optimize a CBM policy as well?Choose [yes] or [no]')

        try:
            x = cond_opt in choices
            count = 0

        except x is False:
            print("Invalid input.")

    student_nr = "s5014670"  # Add your student number, same as in data file names
    data_path = "C:/Users/Modda/PycharmProjects/pythonProject11/Maintainance_data/"  # Example: 'C:/Documents/AssetManagement/Assignment/'
    machine_name = 1
    machine_data = pd.read_csv(f'{data_path}{student_nr}-Machine-{machine}.csv')
    prepared_data = data_preparation(machine_data)

    # Kaplan-Meier estimation
    KM_data = create_kaplanmeier_data(prepared_data)
    MTBF_KM = meantimebetweenfailure_KM(KM_data)
    print('The MTBF-KaplanMeier is: ', MTBF_KM)

    # Weibull fitting
    lamb_val, kap_val = fit_weibull_distribution(prepared_data)
    print(lamb_val)
    print(kap_val)
    weibull_data = create_weibull_curve_data(prepared_data, lamb_val, kap_val)
    MTBF_weibull = meantimebetweenfailure_weibull(lamb_val, kap_val)
    print('The MTBF-Weibull is: ', MTBF_weibull)

    # Visualization
    visualization(KM_data, weibull_data)  # Age-based maintenance optimization
    PM_cost = float(pom)
    CM_cost = float(com)
    best_age, best_cost_rate = create_cost_data(prepared_data, lamb_val, kap_val, PM_cost, CM_cost)
    print('The optimal maintenance age is ', best_age)
    print('The best cost rate is ', best_cost_rate)

    # Condition Based Maintenance
    condition_data = pd.read_csv(f'{data_path}{student_nr}-Machine-3-condition-data.csv')
    prepared_condition_data = CBM_data_preparation(condition_data)

    # get the last observation where machine is not repaired
    max_observation = prepared_condition_data.loc[prepared_condition_data['Condition'].idxmax()]

    # extract the condition-level at which machine breaks down
    failure_level = max_observation['Condition']

    CBM_cost_rate, CBM_threshold = CBM_create_cost_data(prepared_condition_data, PM_cost, CM_cost, failure_level)
    print('The optimal cost rate under CBM is ', CBM_cost_rate)
    print('The optimal CBM threshold is ', CBM_threshold)

    return KM_data


nb = run_analysis()
