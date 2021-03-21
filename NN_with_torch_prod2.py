import torch
import torch.nn as nn
import torch.cuda as cuda
import extractinputs as ei
import support_functions as sf
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import csv
import time


torch.manual_seed(2020)
cuda.manual_seed(2020)
np.random.seed(2020)
os.chdir(os.getcwd())
path = os.getcwd()
device = torch.device("cuda:0" if cuda.is_available() else "cpu")

os.chdir(os.getcwd())
path = os.getcwd()

# Import the DMS dataset
data = pd.read_csv(path + '/Input/DMS_CCS_ML_Dataset_Class.csv')
data.head()

#Un-normalized data
inputs_raw = data.drop(labels=['Combined_CCS', 'Compound'], axis = 'columns')
target = data['Combined_CCS']
names = data['Compound']

#Endcode strings to numerical values
le_Superclass = LabelEncoder()
le_Class = LabelEncoder()

inputs_raw['Superclass_n'] = le_Superclass.fit_transform(inputs_raw['Superclass'])
inputs_raw['Class_n'] = le_Class.fit_transform(inputs_raw['Class'])

inputs = inputs_raw.drop(['Superclass', 'Class'], axis='columns').to_numpy(dtype=float)
target = target.to_numpy(dtype=float).reshape((len(target),1))
names = names.to_numpy(dtype='O').reshape((len(target),1))

def run_NN(input_array,output_array,name_array,out_file_name,filtertype,extfilter,modelsplit = 0.7,ext_splt = 0,runs=10):
    external_list = []
    ext_avg = []
    list_ext_rmse = []
    temp = []
    list_ext_std = []
    list_ext_drmse = []
    
    count=0
    for run in range(runs):
        if extfilter == "B":
            temp = sf.getbiasedlist(output_array,temp,ext_splt)
            external_list.append(temp)
            temp = []
        if extfilter == "N":
            temp = sf.getbiasedlist(output_array,temp,ext_splt)
            external_list.append(temp)
            temp = []
    '''
    out_min = min(output_array)
    out_max = max(output_array)
    output_array = (output_array-out_min) / (out_max-out_min)
    '''
    model = nn.Sequential(nn.Linear(11,18), nn.ReLU(),\
                nn.Linear(18,18), nn.ReLU(),\
                nn.Linear(18,18), nn.ReLU(),\
                nn.Linear(18,18), nn.ReLU(),\
                nn.Linear(18,1), nn.ReLU()).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()},path + "/Output/model_init2.pt")
    while count != runs:
        check_point = torch.load(path+"/Output/model_init2.pt")
        model.load_state_dict(check_point['model_state_dict'])
        optimizer.load_state_dict(check_point['optimizer_state_dict'])
        training_array, training_array_y, test_array, test_array_y, training_names, test_names, test_set_index, train_set_index = sf.split_data(input_array,output_array,name_array,filtertype,training_test_split=modelsplit,external_split=external_list[count])
        training_array = torch.FloatTensor(training_array).to(device)
        test_array = torch.FloatTensor(test_array).to(device)
        training_array_y = torch.FloatTensor(training_array_y).to(device)
        test_array_y = torch.FloatTensor(test_array_y).to(device)
        list_of_test_std = []
        list_of_train_std = []
        list_of_avg_test_error = []
        list_of_avg_training_error = []
        list_of_train_rmse = []
        list_of_train_drmse = []
        list_of_test_rmse = []
        list_of_test_drmse = []
        train_count = 0
        while train_count != 20:
            start_time = time.time()
            #each set is a different split
            training_array, training_array_y, test_array, test_array_y, training_names, test_names, test_set_index, train_set_index = sf.split_data(input_array,output_array,name_array,filtertype,training_test_split=modelsplit,external_split=external_list[count])
            #Converts the inputs arrays to available hardware into torch tensors
            training_array = torch.from_numpy(training_array).float().to(device)
            test_array = torch.from_numpy(test_array).float().to(device)
            training_array_y = torch.from_numpy(training_array_y).float().to(device)
            test_array_y = torch.from_numpy(test_array_y).float().to(device)
            for epoch in range(1500):
                model.train()
                y_pred = model(training_array)
                loss = loss_fn(y_pred, training_array_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            train_out = model(training_array)
            # test_out = model(test_array)

            training_array = training_array.detach().cpu().numpy()
            # test_array = test_array.detach().cpu().numpy()
            train_out = train_out.detach().cpu().numpy()
            training_array_y = training_array_y.detach().cpu().numpy()
            # test_out = test_out.detach().cpu().numpy()
            # test_array_y = test_array_y.detach().cpu().numpy()

            train_table = sf.format_table_for_print(training_names, training_array[:,0:1], train_out, training_array_y)
            # test_table = sf.format_table_for_print(test_names, test_array[:,0:1],test_out, test_array_y)
            train_max_error, train_percent_error, train_average_percent_error, train_std,train_max_error_row,train_max_percent_error = sf.print_info(train_table, train_out, training_array_y)
            # test_max_error, test_percent_error, test_average_percent_error, test_std, test_max_error_row,test_max_percent_error= sf.print_info(test_table, test_out, test_array_y) 

            train_rmse, train_drmse = sf.mean_square_error(train_percent_error)
            # test_rmse, test_drmse = sf.mean_square_error(test_percent_error)
            if train_average_percent_error >=4:
                pass
            else:
                # list_of_avg_test_error.append(test_average_percent_error)
                # list_of_test_std.append(test_std)
                list_of_avg_training_error.append(train_average_percent_error)
                list_of_train_std.append(train_std)
                list_of_train_rmse.append(train_rmse)
                list_of_train_drmse.append(train_drmse)
                # list_of_test_rmse.append(test_rmse)
                # list_of_test_drmse.append(test_drmse)

                sf.write_value(train_out,training_array_y,train_max_error, train_average_percent_error, train_max_error_row, train_max_percent_error,out_file_name,"train",training_names,train_percent_error)
                # sf.write_value(test_out,test_array_y,test_max_error, test_average_percent_error, test_max_error_row, test_max_percent_error,out_file_name,"test",test_names,test_percent_error)
                train_count+=1
                #print("Time taken for one iteration is " + str(time.time()-start_time))
        if train_average_percent_error >= 4:
            pass
        else:
            ext_test, ext_test_y, ext_name = sf.external_val(input_array,output_array,name_array,external_list[count]) 
            ext_test = torch.from_numpy(ext_test).float().to(device)
            model.eval()
            ext_out = model(ext_test)

            ext_test = ext_test.detach().cpu().numpy()
            ext_out = ext_out.detach().cpu().numpy()
            '''
            ext_out = ext_out*(out_max-out_min) + out_min
            ext_test = ext_test*(out_max-out_min) +out_min
            ext_test_y = (ext_test_y)*(out_max-out_min)+out_min
            '''
            count +=1
            train_count = 0
            
            ext_test_table = sf.format_table_for_print(ext_name, ext_test[:,0:1], ext_out, ext_test_y)
            ext_max_error, ext_percent_error, ext_average_percent_error, ext_std, ext_max_error_row, ext_max_percent_error = sf.print_info(ext_test_table, ext_out, ext_test_y) 
            sf.write_value(ext_out,ext_test_y,ext_max_error,ext_average_percent_error, ext_max_error_row, ext_max_percent_error,out_file_name,"ext",ext_name,ext_percent_error)
            
            ext_avg.append(ext_average_percent_error)
            list_ext_std.append(ext_std)

            ext_rmse, ext_drmse = sf.mean_square_error(ext_percent_error)
            list_ext_rmse.append(ext_rmse)
            list_ext_drmse.append(ext_drmse)

            outputfile = open(out_file_name,"a",newline='')
            writer = csv.writer(outputfile)
            writer.writerow(list_of_avg_training_error)
            writer.writerow(list_of_train_std)
            writer.writerow(list_of_avg_test_error)
            writer.writerow(list_of_test_std)
            line0 = "Average model set error over " + str(len(list_of_avg_training_error)) +" sample is," + str(np.average(list_of_avg_training_error)) + "\n"
            line1 = "Standard deviation for model set over " + str(len(list_of_avg_training_error)) + " is, " + str(np.std(list_of_avg_training_error,ddof=1)) +"\n"
            line = "Average validation set error over " + str(len(list_of_avg_test_error)) +" sample is," + str(np.average(list_of_avg_test_error)) + "\n"
            line2 = "Standard deviation for validation set over " + str(len(list_of_avg_test_error)) + " is, " + str(np.std(list_of_avg_test_error,ddof=1)) +"\n"
            outputfile.writelines(line0)
            outputfile.writelines(line)
            outputfile.writelines(line1)
            outputfile.writelines(line2)
            outputfile.close()

    outputfile = open(out_file_name,"a",newline='')
    outputfile.writelines("\n")
    writer = csv.writer(outputfile)
    writer.writerow(ext_avg)
    writer.writerow(list_ext_std)
    writer.writerow(list_ext_rmse)
    writer.writerow(list_ext_drmse)
    outputfile.writelines("Average external validation set error over " +str(count) + " sample is,"+ str(np.average(ext_avg))+"\n")
    outputfile.writelines("Standard deviation for external validation set error over " +str(count) + " sample is,"+ str(np.std(ext_avg,ddof=1))+"\n")
    outputfile.writelines("Root mean Square error external validation set over " +str(count) + " sample is,"+ str(np.average(list_ext_rmse))+"\n")
    outputfile.writelines("Root mean Square error deviation for external validation set error over " +str(count) + " sample is,"+ str(np.std(list_ext_rmse,ddof=1))+"\n")
    outputfile.close()

print("starting run")
run_NN(inputs,target,names,path + "/Output/Updated 5 layer biased filter 9505 mod_ext with all Normal Training 1500 optimize with superclass label.csv",filtertype="N",extfilter="B",modelsplit=1,ext_splt=0.95,runs=20)