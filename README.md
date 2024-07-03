# Water-dimer

To create and Fourier network run with julia Run_water_dimer.jl with params_gen.txt piped into it.
Example for running in terminal: julia Run_water_dimer <params_gen.txt >output.txt &

To adjust the parameters edit params_gen.txt, each line of params_gen.txt is commented on what it does.

The code ouputs into a folder specified in params_gen.txt a mixture of jld2 and text files
  a. The Fourier model
  b. The data it was trained on
  c. Testing data (exclusive from training)
  d. scaling done to preprocess the data, can be used to convert to orginal units
  e. Parameters a comment file containing all parameters related to model

To used the Show_Fourier_model.ipynb just change the string path (second section) to the name of the folder that contains the run you want to look at.

Every file should be in the same folder. 
