import utils
import pandas as pd
from etl import read_csv
from etl import create_features
from etl import save_svmlight
from models_partc import *
from sklearn.metrics import *
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold, ShuffleSplit

#Note: You can reuse code that you wrote in etl.py and models.py and cross.py over here. It might help.
# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

'''
You may generate your own features over here.
Note that for the test data, all events are already filtered such that they fall in the observation 
window of their respective patients. Thus, if you were to generate features similar to those you 
constructed in code/etl.py for the test data, all you have to do is aggregate events for each patient.

IMPORTANT: Store your test data features in a file called "test_features.txt" where each line has the
patient_id followed by a space and the corresponding feature in sparse format.
Eg of a line:
60 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514
Here, 60 is the patient id and 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 
3005:0.367953 3049:0.013514 is the feature for the patient with id 60.

Save the file as "test_features.txt" and save it inside the folder deliverables

input:
output: X_train,Y_train,X_test
'''

def my_features():
    #TODO: complete this

    X_train, Y_train = utils.get_data_from_svmlight("../deliverables/features_svmlight_for_my_model.train")

    # get events and feature_map from test folder
    events, _, feature_map = read_csv('../data/test/')

    # get mortality from train folder
    mortality = pd.DataFrame({'patient_id':[], 'timestamp':[], 'label':[]})
            # _, mortality, _ = read_csv('../data/train/')

    # # create feature and mortality dictionary
    patient_features, _ = create_features(events, mortality, feature_map)
    
    # # save to svmlight format
    def save_svmlight(patient_features, op_file):

        deliverable = open(op_file, 'wb+')

        def dict_pair_to_string(ls):
            return ' '.join(( "%d:%f" % (feature, float(value)) for feature, value in ls))

        for patient, featureList in patient_features.iteritems():
            featureList = pd.DataFrame(featureList).sort_values(0)

            featureList = featureList.values.tolist()

            deliverable.write("{} {} \n".format(str(patient),
                                          dict_pair_to_string(featureList)))
        deliverable.close()


    save_svmlight(patient_features, '../deliverables/test_features.txt')
    
    X_test,  Y_test  = utils.get_data_from_svmlight("../deliverables/test_features.txt")

    return X_train,Y_train,X_test

'''
You can use any model you wish.

input: X_train, Y_train, X_test
output: Y_pred
'''

def my_classifier_predictions(X_train,Y_train,X_test, N_SIZE = 45, MAX_DEPTH = 4):
    #TODO: complete this

    # # use nn
    # clf = MLPClassifier(solver='lbfgs', alpha=1e-1, hidden_layer_sizes=(20, 20, 10))

    # use adaboost with DecisionTreeClassifier or RandomForestClassifier
    clf = AdaBoostClassifier(
        base_estimator = RandomForestClassifier(max_depth = MAX_DEPTH, 
                            criterion= "gini", n_estimators = 20), 
        n_estimators = N_SIZE)
    # ,random_state = 1

    clf.fit(X_train, Y_train)

    # X_test is blind test data, we do not know Y_test's label
    Y_pred = clf.predict(X_test)
    # display_metrics("AdaBoost + RandomForest Classifier training stats",Y_pred,Y_train)

    ######
    # validation set
    X_validate, Y_validate = utils.get_data_from_svmlight("../data/features_svmlight.validate")
    print "N_SIZE= %3d MAX_DEPTH= %3d  AUC= %.3f" \
            % (N_SIZE, MAX_DEPTH, roc_auc_score(Y_validate, clf.predict(X_validate)))
    
    # display_metrics("AdaBoost + RandomForest Classifier use given testing stats",
    #                 clf.predict(X_validate), Y_validate)
    
    return Y_pred

def main():
    X_train, Y_train, X_test = my_features()
    Y_pred = my_classifier_predictions(X_train,Y_train,X_test)

    #####
    # testing for which parameter is better
    
    # for n_size in xrange(20,100,5):
    #     for max_depth in xrange(2,10,2):
    #         Y_pred = my_classifier_predictions(X_train,Y_train,X_test, 
    #                                             N_SIZE = n_size, MAX_DEPTH = max_depth)

    #####

    utils.generate_submission("../deliverables/test_features.txt",Y_pred)
    # The above function will generate a csv file of (patient_id,predicted label) 
    # and will be saved as "my_predictions.csv" in the deliverables folder.

if __name__ == "__main__":
    main()
