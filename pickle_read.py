import pickle
output_folder = 'C:/Users/Rudy/Desktop/DL_models/'
subfolder = "net_type/unet/"

file_name = output_folder + subfolder + "new_pred_list.pkl"
open_file = open(file_name, "rb")
new_list = pickle.load(open_file)
open_file.close()


file_name = output_folder + subfolder + "old_pred_list.pkl"
open_file = open(file_name, "rb")
old_list = pickle.load(open_file)
open_file.close()


file_name = output_folder + subfolder + "new_but_old_pred_list.pkl"
open_file = open(file_name, "rb")
new_but_old_list = pickle.load(open_file)
open_file.close()