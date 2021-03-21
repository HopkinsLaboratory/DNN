#supporting functions are located here as I do not want to make the neural network code to be too long

#importing randint to save space
import numpy as np
import csv
import os
from scipy.stats import norm

#Generate a list of random index to split data
os.chdir(os.getcwd())
path = os.getcwd()

def gen_rand_int_list(empty_index_list, actual_data_set, split_percent,external_split=[]):
    test_index_list_len = round((len(actual_data_set)-len(external_split))*(1-split_percent))
    #print("length of mass list is", len(list_of_mass))
    #print("length of testlist should be", test_index_list_len)

    while test_index_list_len != len(empty_index_list):
        rand_num = np.random.randint(0,len(actual_data_set)-1)
        if rand_num in empty_index_list:
            continue
        elif rand_num in external_split:
            continue
        else:
            empty_index_list.append(rand_num)
    return empty_index_list

def gen_test_list(criteria_list, empty_index_list, split_percent,external_split=[]):
    if external_split != []:
        temp = []
        newCriteria_list = []
        temp = gen_remaining_list(temp,external_split,criteria_list)
        for new in temp:
            newCriteria_list.append(criteria_list[new])
        mode = norm.fit(newCriteria_list)[0]
        std = np.std(newCriteria_list,ddof=1)
    else:
        mode = norm.fit(criteria_list)[0]
        std = np.std(criteria_list,ddof=1)
    
    #need to change std multiplier when you have more element in test set
    if split_percent>=0.7:
        upperbound = mode + std*1
        lowerbound = mode - std*1
    elif split_percent >= 0.5:
        upperbound = mode + std*1.5
        lowerbound = mode - std*1.5
    elif split_percent >= 0.3:
        upperbound = mode + std*2
        lowerbound = mode - std*2
    test_index_list_len = round((len(criteria_list)-len(external_split))*(1-split_percent))
    while test_index_list_len != len(empty_index_list):
        rand_num = np.random.randint(0,len(criteria_list)-1)
        criteria = criteria_list[rand_num][0]
        if  lowerbound <= criteria <= upperbound:
            if rand_num in empty_index_list:
                continue
            elif rand_num in external_split:
                continue
            else:
                empty_index_list.append(rand_num)
    return empty_index_list

def read_test_list(index,filename):
    csvrows = []
    csvreader = csv.reader(open(filename))
    for row in csvreader:
        csvrows.append(row)
    return csvrows[index]

#Generate the list for the rest of the indicies
def gen_remaining_list(empty_index_list,generated_index_list,actual_data_set, external_split=[]):
    for i in range(int(round(len(actual_data_set)))):
        if i in external_split:
            continue
        elif i in generated_index_list:
            continue
        else:
            empty_index_list.append(i)
            
    return empty_index_list

def split_data(x,y,name,sample_type,external_split=[],index=-1,training_test_split=0.7,support_query_split=0):
        index_for_test_set, index_for_training_set = [],[]
        training_set, test_set = np.empty(x.shape[1]),np.empty(x.shape[1])
        training_set_y, test_set_y, name_training_set, name_test_set = np.zeros(y.shape[1]),np.zeros(y.shape[1]),np.zeros(y.shape[1]),np.zeros(y.shape[1])

        if sample_type == "B": #B for biased sample
            index_for_test_set = gen_test_list(y,index_for_test_set,training_test_split,external_split=external_split)
            index_for_training_set = gen_remaining_list(index_for_training_set,index_for_test_set,x,external_split=external_split)
        elif sample_type == "N": #N for normal sample
            index_for_test_set = gen_rand_int_list(index_for_test_set,x,training_test_split,external_split=external_split)
            index_for_training_set = gen_remaining_list(index_for_training_set,index_for_test_set,x,external_split=external_split)
        elif sample_type == "PN": #P for Pre determined sampling
            pre_list = read_test_list(index, path+"/Input/Normal List.csv")
            for num in range(len(pre_list)):
                pre_list[num] = int(pre_list[num])
            index_for_test_set = pre_list
            index_for_training_set = gen_remaining_list(index_for_training_set,index_for_test_set,x,external_split=external_split)
        elif sample_type == "PB":
            pre_list = read_test_list(index,path+"/Input/biased List.csv")
            for num in range(len(pre_list)):
                pre_list[num] = int(pre_list[num])
            index_for_test_set = pre_list
            index_for_training_set = gen_remaining_list(index_for_training_set,index_for_test_set,x,external_split=external_split)
        else:
            pass
        #make sure vstack is a tuple 
        for Training_indicies in index_for_training_set:
            training_set = np.vstack((training_set,x[Training_indicies]))
            training_set_y = np.vstack((training_set_y,y[Training_indicies]))
            name_training_set = np.vstack((name_training_set,name[Training_indicies]))

        for Testing_indicies in index_for_test_set:
            test_set = np.vstack((test_set,x[Testing_indicies]))
            test_set_y = np.vstack((test_set_y,y[Testing_indicies]))
            name_test_set = np.vstack((name_test_set,name[Testing_indicies]))
        
        #enable support and query set change the support query split value 
        if support_query_split == 0:
            return training_set[1:], training_set_y[1:], test_set[1:], test_set_y[1:], name_training_set[1:], name_test_set[1:], index_for_test_set, index_for_training_set

        else:
            index_for_support,index_for_query = [],[]
            support_set, query_set = np.empty(x.shape[1]),np.empty(x.shape[1])
            support_set_y, query_set_y = np.empty(y.shape[1]),np.empty(y.shape[1])

            #index_for_support = gen_rand_int_list(index_for_support,training_set,support_query_split)
            #index_for_query = gen_remaining_list(index_for_query,index_for_support,training_set,(1-support_query_split))
            
            for support_indicies in index_for_support:
                support_set = np.vstack((training_set[support_indicies]))
                support_set_y = np.vstack((training_set_y[support_indicies]))
            
            for query_indicies in index_for_query:
                query_set = np.vstack((training_set[query_indicies]))
                query_set_y = np.vstack((training_set_y[query_indicies]))
            
            return test_set[1:], test_set_y[1:], support_set[1:], support_set_y[1:], query_set[1:], query_set_y[1:]

def external_val(x,y,name,listofint):
    test_set = np.empty(x.shape[1])
    test_set_y,  name_test_set = np.zeros(y.shape[1]),np.zeros(y.shape[1])
    for Testing_indicies in listofint:
        test_set = np.vstack((test_set,x[Testing_indicies]))
        test_set_y = np.vstack((test_set_y,y[Testing_indicies]))
        name_test_set = np.vstack((name_test_set,name[Testing_indicies]))
    return test_set[1:], test_set_y[1:], name_test_set[1:]

def format_table_for_print(name_list, weight_input, nn_value, obs_value):
    header = np.array(["Compound name","Weight input","NN Y value", "Target Y value", "Error"],dtype="object")
    table = np.empty([len(nn_value),5],dtype="object")
    table[:,0:1] = name_list
    table[:,1:2] = weight_input
    table[:,2:3] = nn_value
    table[:,3:4] = obs_value
    error = nn_value - obs_value
    table[:,4:5] = error
    for line in table:
        header = np.vstack((header,line))
    return header

def print_info(table,nn_value,obs_value):
    error = nn_value - obs_value
    max_error = max(np.absolute(error))
    location = np.where(np.absolute(error) == max_error)
    row_location = location[0][0] + 1
    percent_error = error/obs_value
    average_percent_error = np.average(np.absolute(percent_error)*100)
    error_dev = np.std(percent_error,ddof=1)
    max_error_row = table[row_location]
    max_percent_error = 1
    
    return max_error, percent_error, average_percent_error, error_dev, max_error_row, max_percent_error 

def write_value(output,mobcal,max_error, average_percent_error, max_error_row, max_percent_error,filename = None,tort = "train",index_table = None,error_table = None,average_training_error_percent = 0 ,avg_error_list = 0):
    try:
        outputfile = open(filename, "a", newline='')
        if tort == "train":
            outputfile.writelines("The following is model set \n")
        elif tort == "test":
            outputfile.writelines("The following is validation set \n")
        elif tort =="ext":
            outputfile.writelines("The following is external validation set\n")
    except FileNotFoundError:
        outputfile = open(filename,"w",newline='')
        if tort == "train":
            outputfile.writelines("The following is model set")
        elif tort == "test":
            outputfile.writelines("The following is validation set")
        elif tort =="ext":
            outputfile.writelines("The following is external validation set")
    if index_table is None:
        pass
    else:
        outputfilewrite = csv.writer(outputfile)
        indexlist =[]
        for item in index_table:
            indexlist.append(item[0])
        outputfilewrite.writerow(indexlist)
        mobcal = mobcal.T
        output = output.T
        error_table = error_table.T
        for lines in range(len(output)):
            outputfile.writelines("ML Prediction:"+"\n")
            outputfilewrite.writerow(output[lines])
        lines = 0
        for lines in range(len(mobcal)):
            outputfile.writelines("Mobcal Prediction:"+"\n")
            outputfilewrite.writerow(mobcal[lines])    
        lines = 0
        for lines in range (len(error_table)):
            outputfile.writelines("Error:"+"\n")
            outputfilewrite.writerow(error_table[lines])
        outputfile.writelines("\n")
    stdev = np.std(np.absolute(error_table))
    avg_error_of_all = str(np.average(avg_error_list))
    error = ((np.absolute(output-mobcal)/mobcal)*100)**2
    mse = error.mean()
    rmse = np.sqrt(mse)
    rmsed = np.sqrt(np.std(error,ddof=1))
    if average_training_error_percent == 0:
        pass
    else:
        outputfile.writelines("Average Training Set error is, "+ str(average_training_error_percent)+"%\n")
    outputfile.writelines("Average error,"+str(average_percent_error)+"%\n")
    if avg_error_list == 0:
        pass
    else:
        outputfile.writelines("Average error over " + str(len(avg_error_list)) + " is " + avg_error_of_all +"\n")
    outputfile.writelines("Deviation," + str(stdev)+"\n")
    outputfile.writelines("root mean square error," + str(rmse)+"\n")
    outputfile.writelines("root mean square error deviation," + str(rmsed)+"\n")
    outputfile.writelines("\n")
    outputfile.close()

def split_extracted(extracted_data):
    data_array = np.array(extracted_data[1:])
    input_array = np.array(data_array[:,:-2],dtype=float)
    output_array = np.array(data_array[:,-2:-1],dtype=float)
    name_array = data_array[:,-1:]

    return data_array, input_array, output_array, name_array

def getbiasedlist(criteria_list, empty_index_list, split_percent):
    mode = norm.fit(criteria_list)[0]
    std = np.std(criteria_list,ddof=1)

    #need to change std multiplier when you have more element in test set
    if split_percent>=0.7:
        upperbound = mode + std*1
        lowerbound = mode - std*1
    elif split_percent >= 0.5:
        upperbound = mode + std*1.5
        lowerbound = mode - std*1.5
    elif split_percent >= 0.3:
        upperbound = mode + std*2
        lowerbound = mode - std*2

    test_index_list_len = round(len(criteria_list)*(1-split_percent))
    while test_index_list_len != len(empty_index_list):
        rand_num = np.random.randint(0,len(criteria_list)-1)
        critieria = criteria_list[rand_num][0]
        if  lowerbound <= critieria <= upperbound:
            if rand_num in empty_index_list:
                continue
            else:
                empty_index_list.append(rand_num)
    return empty_index_list

def getnormallist(actual_data_set,empty_index_list, split_percent):
    test_index_list_len = round(len(actual_data_set)*(1-split_percent))
    #print("length of mass list is", len(list_of_mass))
    #print("length of testlist should be", test_index_list_len)

    while test_index_list_len != len(empty_index_list):
        rand_num = np.random.randint(0,len(actual_data_set)-1)
        if rand_num in empty_index_list:
            continue
        else:
            empty_index_list.append(rand_num)
    return empty_index_list

def mean_square_error(errorlist):
    mse = (errorlist**2).mean()
    rmse = np.sqrt(mse)
    rmsed = np.sqrt(np.std(errorlist,ddof=1))
    return rmse, rmsed