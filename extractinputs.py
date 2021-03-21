#Import the native csv module for dealing with the data file that is in csv
import csv

#extracting inputs from the parameter files here. 
#Returns a dictionary that contains all the parameters from the input
def extractparameters(input_parameter_filename):
    parameterfile = open(input_parameter_filename,"r")
    parameters_infile = parameterfile.readlines()
    parameterfile.close()
    trimmed_parameters = []
    parameters = {}
    #reading data from the files and cleaning up all the spaces and line spacing
    for lines in parameters_infile:
        if lines[1] == "*":
            for i in range (len(lines)):
                if lines[i] == "=":
                    trimmed_parameters.append(lines.strip()[i+1:])
                else:
                    pass
        else:
            pass
    #assign the values to a dictionary for quick access 
    parameter_names = ["learner", "cact", "n_layer", "nodes(2)","nodes(3)", "nodes(4)", "testsplit", "num_model", "wt_bt_file","u_s_file", "alpha", "iseed", "tr_result", "te_result","cttmax", "delcttmax", "delcttmin", "itmax", "valfile"]
    for names in parameter_names:
        for parameter_value in trimmed_parameters:
            parameters[names] = parameter_value
            trimmed_parameters.remove(parameter_value)
            break
    return parameters


#reads the csv datafile, extracts the data into an array
def extractdata(name_of_input_csv):
    csvrows = []
    csvreader = csv.reader(open(name_of_input_csv))
    for row in csvreader:
        csvrows.append(row)
    return csvrows
