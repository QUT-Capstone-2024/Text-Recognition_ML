Whilst this is under local development each time you do a fresh pull you will need to:

Change this var in the app.py before running the server or you will get an error as the path won't point towards the 
tokenizermodel_path = "C:/Users/SLHan/OneDrive/Desktop/Capstone/Language-Model/Language-Model/LLM_RoBERTa/trained_model"

You will also want to chnage this var in the input_prototype.py if you plan to use this for testing rather than the API and Postman
model_path = model_path = r'C:\Users\SLHan\OneDrive\Desktop\Capstone\Language-Model\Text-Recognition_ML\LLM_RoBERTa\trained_model'

There is a collection in the group Postman for this api to test functionality once it is running. 

To start the serevr open a terminal and enter:
./app.py (for powershell whcih is what VSC uses as standard)

If you get errors when running this you have most likely forgotten to update the paths above correctly or you have not got all the required packages installed.
Read the errors and do:
pip install <package-name> 
for each missing package and try again.

