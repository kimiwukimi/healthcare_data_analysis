import utils
import collections
import pandas as pd
import numpy as np

# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    
    '''
    TODO: This function needs to be completed.
    Read the events.csv, mortality_events.csv and event_feature_map.csv files into events, 
    mortality and feature_map.
    
    Return events, mortality and feature_map
    '''

    #Columns in events.csv - patient_id,event_id,event_description,timestamp,value
    events = pd.read_csv(filepath + 'events.csv')

    #Columns in mortality_event.csv - patient_id,timestamp,label
    # try except is for my_model's exception
    try:
        mortality = pd.read_csv(filepath + 'mortality_events.csv')
        mortality.timestamp = pd.to_datetime(mortality.timestamp)
    except:
        mortality = None

    # convert timestamp to datatime for later use in different functions
    events.timestamp = pd.to_datetime(events.timestamp)
    

    #Columns in event_feature_map.csv - idx,event_id
    feature_map = pd.read_csv(filepath + 'event_feature_map.csv')

    # check here
    # print  len(events), len(mortality), len(feature_map)
    return events, mortality, feature_map


def calculate_index_date(events, mortality, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 a

    Suggested steps:
    1. Create list of patients alive ( mortality_events.csv only contains information about patients deceased)
    2. Split events into two groups based on whether the patient is alive or deceased
    3. Calculate index date for each patient
    
    IMPORTANT:
    Save indx_date to a csv file in the deliverables folder named as etl_index_dates.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, indx_date.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)

    Return indx_date
    '''
    
    # for alive
    alive = events[~events.patient_id.isin(mortality.patient_id)]
    alive = alive[['patient_id', 'timestamp']]

    # get lates date of events of every patients
    alive_time = alive.groupby(["patient_id"]).max()

    # set index to one column (becuz group by make patient_id be index)
    alive_time.reset_index(level=0, inplace=True) 

    # for dead
    dead_time = mortality[['patient_id', 'timestamp']]
    dead_time.timestamp += pd.offsets.Day(-30) # set indx time to 30 days before death

    # concat dead and alive
    indx_date = pd.concat([dead_time, alive_time])

    # rename timestamp to indx_data
    indx_date.rename(columns={'timestamp': 'indx_date'}, inplace=True)

    # write df to csv (as instructed)
    indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', 
                    columns=['patient_id', 'indx_date'], index=False)

    return indx_date


def filter_events(events, indx_date, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 b

    Suggested steps:
    1. Join indx_date with events on patient_id
    2. Filter events occuring in the observation window(IndexDate-2000 to IndexDate)
    
    
    IMPORTANT:
    Save filtered_events to a csv file in the deliverables folder named as etl_filtered_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)

    Return filtered_events
    '''
    
    # left merge 
    df = pd.merge(events, indx_date, how='left', on='patient_id')

    # filter time must be within [indx_data-2000days, indx_date]
    filtered_events = df[(df.timestamp >= (df.indx_date + pd.offsets.Day(-2000))) 
                        & (df.timestamp <= df.indx_date)]

    # select 3 columns (as instructed)
    filtered_events = filtered_events[['patient_id', 'event_id', 'value']]
    
    # save to csv (as instructed)
    filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', 
                            columns=['patient_id', 'event_id', 'value'], index=False)

    return filtered_events

def aggregate_events(filtered_events_df, mortality_df,feature_map_df, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 c

    Suggested steps:
    1. Replace event_id's with index available in event_feature_map.csv
    2. Remove events with n/a values
    3. Aggregate events using sum and count to calculate feature value
    4. Normalize the values obtained above using min-max normalization(the min value will be 0 in all scenarios)
    
    
    IMPORTANT:
    Save aggregated_events to a csv file in the deliverables folder named as etl_aggregated_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header .
    For example if you are using Pandas, you could write: 
        aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)

    Return filtered_events
    '''

    # 1.replace event_id with number coding
    # joined = filtered_events_df.join(feature_map_df.set_index('event_id'),
    #                                  on='event_id')
    # filtered_events_df['feature_id'] = joined['idx']
    filtered_events_df = pd.merge(filtered_events_df, feature_map_df, how='left', on='event_id')
    filtered_events_df.rename(columns={'idx':'feature_id'}, inplace=True)

    # 2.drop na
    filtered_events_df = filtered_events_df.dropna()
    # filtered_events_df = filtered_events_df[~filtered_events_df.isnull().any(axis=1)]

    # 3. split into 2, process, then combine
    merged = filtered_events_df
    cols = ['patient_id', 'event_id', 'feature_id']

    lab_df = merged[merged.event_id.str.contains('LAB')]
    lab_df = lab_df.groupby(cols).patient_id.count()    
    # !!! BIG bug here, need to assign to lab_df. group by returns a groupby object
    # 4 hours of debugging 1/15/2018

    diag_df = merged[np.logical_or(merged.event_id.str.contains('DIAG'), 
                                  merged.event_id.str.contains('DRUG'))]
    diag_df = diag_df.groupby(cols).value.sum()

    # combine
    valued_df = pd.concat([lab_df, diag_df]).reset_index()

    valued_df.columns = cols + ['feature_value']
    valued_df = valued_df[['patient_id', 'feature_id', 'feature_value']]

    # 4. normaliza

    # valued_df = aggregated_events
    pivoted = valued_df.pivot(index='patient_id', columns='feature_id', values='feature_value')
    normed = pivoted / pivoted.max()
    # normed = (pivoted - pivoted.min())
    # normed = normed / (normed.max() - normed.min())

    normed = normed.reset_index()
    aggregated_events = pd.melt(normed, id_vars='patient_id',
                                value_name='feature_value').dropna()

    aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', 
                            columns=['patient_id', 'feature_id', 'feature_value'], index=False)
    return aggregated_events

def create_features(events, mortality, feature_map):
    
    deliverables_path = '../deliverables/'

    #Calculate index date
    indx_date = calculate_index_date(events, mortality, deliverables_path)

    #Filter events in the observation window
    filtered_events = filter_events(events, indx_date,  deliverables_path)
    
    #Aggregate the event values for each patient 
    aggregated_events = aggregate_events(filtered_events, mortality, feature_map, deliverables_path)

    '''
    TODO: Complete the code below by creating two dictionaries - 
    1. patient_features :  Key - patient_id and value is array of tuples(feature_id, feature_value)
    2. mortality : Key - patient_id and value is mortality label
    '''

    patient_features = collections.defaultdict(list)
    for index, row in aggregated_events.iterrows():
        key = row.patient_id
        value = (row.feature_id, row.feature_value)
        patient_features[key].append(value)
    
    for k, v in patient_features.iteritems():
        v.sort(key = lambda x: x[0])

    all_ids = aggregated_events.patient_id.unique() # <type 'numpy.ndarray'>
    dead_ids = set(mortality.patient_id)

    # use above ids to get mortality dict
    mortality = {}
    for id in list(all_ids):
        mortality[id] = 1 if id in dead_ids else 0

    return patient_features, mortality


def save_svmlight(patient_features, mortality, op_file, op_deliverable):
    
    '''
    TODO: This function needs to be completed

    Refer to instructions in Q3 d

    Create two files:
    1. op_file - which saves the features in svmlight format. (See instructions in Q3d for detailed explanation)
    2. op_deliverable - which saves the features in following format:
       patient_id1 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...
       patient_id2 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...  
    
    Note: Please make sure the features are ordered in ascending order, and patients are stored in ascending order as well.     
    '''

    deliverable1 = open(op_file, 'wb')
    deliverable2 = open(op_deliverable, 'wb')

    def formatHelper(ls):
        return ' '.join(( "%d:%f" % (feature, float(value)) for feature, value in ls))

    for patient, featureList in patient_features.iteritems():
        featureList = pd.DataFrame(featureList).sort_values(0)

        featureList = featureList.values.tolist()

        deliverable1.write("{} {} \n".format(mortality.get(patient, 0),
                                          formatHelper(featureList)))
        deliverable2.write("{} {} {} \n".format(int(patient),
                                                mortality.get(patient, 0),
                                                formatHelper(featureList)))

    deliverable1.close()
    deliverable2.close()


def main():
    train_path = '../data/train/'
    events, mortality, feature_map = read_csv(train_path)
    patient_features, mortality = create_features(events, mortality, feature_map)
    save_svmlight(patient_features, mortality, '../deliverables/features_svmlight.train', '../deliverables/features.train')

if __name__ == "__main__":
    main()