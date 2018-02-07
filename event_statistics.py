import time
import pandas as pd
import numpy as np

# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    '''
    TODO : This function needs to be completed.
    Read the events.csv and mortality_events.csv files. 
    Variables returned from this function are passed as input to the metric functions.
    '''
    events = pd.read_csv(filepath + 'events.csv')
    mortality = pd.read_csv(filepath + 'mortality_events.csv')
    # print len(events), len(mortality)
    # print mortality.head()
    return events, mortality

def event_count_metrics(events, mortality):
    '''
    TODO : Implement this function to return the event count metrics.
    Event count is defined as the number of events recorded for a given patient.
    '''
    dead = events[events.patient_id.isin(mortality.patient_id)].groupby(['patient_id']).event_id.count()
    alive = events[~events.patient_id.isin(mortality.patient_id)].groupby(['patient_id']).event_id.count()
    # print len(alive)+len(dead)
    # print dead.head()
    # print alive.head()
    # print events.head()

    min_dead_event_count = dead.min()
    max_dead_event_count = dead.max()
    avg_dead_event_count = dead.mean()

    avg_alive_event_count = alive.mean()
    max_alive_event_count = alive.max()
    min_alive_event_count = alive.min()

    # print min_dead_event_count, max_dead_event_count, avg_dead_event_count, min_alive_event_count, max_alive_event_count, avg_alive_event_count
    # (177, 562, 369.5, 238, 1786, 1012.0)
    return min_dead_event_count, max_dead_event_count, avg_dead_event_count, min_alive_event_count, max_alive_event_count, avg_alive_event_count

def encounter_count_metrics(events, mortality):
    '''
    TODO : Implement this function to return the encounter count metrics.
    Encounter count is defined as the count of unique dates on which a given patient visited the ICU. 
    '''
    dead  = pd.merge(events, mortality, on='patient_id')    # try merge, not using indexing method as above
    alive = events[~events.patient_id.isin(dead.patient_id)]

    # print dead.groupby(['patient_id']).head()
    # print dead_unique_time_df.head()
    # print dead.groupby(['patient_id']).head()

    dead_unique_time_df = dead.groupby(['patient_id']).timestamp_x.nunique()
    alive_unique_time_df = alive.groupby(['patient_id']).timestamp.nunique()

    avg_dead_encounter_count = dead_unique_time_df.mean()
    max_dead_encounter_count = dead_unique_time_df.max()
    min_dead_encounter_count = dead_unique_time_df.min()
    avg_alive_encounter_count = alive_unique_time_df.mean()
    max_alive_encounter_count = alive_unique_time_df.max()
    min_alive_encounter_count = alive_unique_time_df.min()
    # assert encounter_count == (5, 14, 9.5, 10, 83, 46.5)
    return min_dead_encounter_count, max_dead_encounter_count, avg_dead_encounter_count, min_alive_encounter_count, max_alive_encounter_count, avg_alive_encounter_count

def record_length_metrics(events, mortality):
    '''
    TODO: Implement this function to return the record length metrics.
    Record length is the duration between the first event and the last event for a given patient. 
    '''
    dead  = events[events.patient_id.isin(mortality.patient_id)]
    alive = events[~events.patient_id.isin(mortality.patient_id)]

    # change col['timestamp'] to datatime
    dead.loc[:,'timestamp'] = pd.to_datetime(dead.loc[:,'timestamp'])
    alive.loc[:,'timestamp'] = pd.to_datetime(alive.loc[:,'timestamp'])

    # use aggregate function to generate new df
    def my_agg(x):
        names = {
            'minTime': x['timestamp'].min(),
            'maxTime': x['timestamp'].max(),
            'range': x['timestamp'].max() - x['timestamp'].min()}

        return pd.Series(names, index=['minTime', 'maxTime', 'range'])

    dead_group = dead.groupby(['patient_id']).apply(my_agg)
    alive_group = alive.groupby(['patient_id']).apply(my_agg)


    avg_dead_rec_len = (dead_group.range.mean() / np.timedelta64(1, 'D')).astype(float)
    min_dead_rec_len = (dead_group.range.min() / np.timedelta64(1, 'D')).astype(float)
    max_dead_rec_len = (dead_group.range.max() / np.timedelta64(1, 'D')).astype(float)
    avg_alive_rec_len = (alive_group.range.mean() / np.timedelta64(1, 'D')).astype(float)
    min_alive_rec_len = (alive_group.range.min() / np.timedelta64(1, 'D')).astype(float)
    max_alive_rec_len = (alive_group.range.max() / np.timedelta64(1, 'D')).astype(float)

    # assert record_length == (4, 633, 318.5, 150, 1267, 708.5)
    return min_dead_rec_len, max_dead_rec_len, avg_dead_rec_len, min_alive_rec_len, max_alive_rec_len, avg_alive_rec_len

def main():
    '''
    You may change the train_path variable to point to your train data directory.
    OTHER THAN THAT, DO NOT MODIFY THIS FUNCTION.
    '''
    # You may change the following line to point the train_path variable to your train data directory
    train_path = '../data/train/'
    # train_path = '../tests/data/statistics/'

    # DO NOT CHANGE ANYTHING BELOW THIS ----------------------------
    events, mortality = read_csv(train_path)

    #Compute the event count metrics
    start_time = time.time()
    event_count = event_count_metrics(events, mortality)
    end_time = time.time()
    print("Time to compute event count metrics: " + str(end_time - start_time) + "s")
    print event_count

    #Compute the encounter count metrics
    start_time = time.time()
    encounter_count = encounter_count_metrics(events, mortality)
    end_time = time.time()
    print("Time to compute encounter count metrics: " + str(end_time - start_time) + "s")
    print encounter_count

    #Compute record length metrics
    start_time = time.time()
    record_length = record_length_metrics(events, mortality)
    end_time = time.time()
    print("Time to compute record length metrics: " + str(end_time - start_time) + "s")
    print record_length
    
if __name__ == "__main__":
    main()
