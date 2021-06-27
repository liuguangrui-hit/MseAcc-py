from datetime import datetime
import os

def save(model,i = 0,name = ""):

    # save the model
    output_path = "saved_models/"
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    print("[INFO] {} - Saving model ...".format(now))
    logdir = output_path + name + '/'
    # try:
    #     os.mkdir(logdir)
    # except OSError as err:
    #     # print("Creation of directory {} failed:{}".format(logdir, err))
    if i == 0 :
        model.save(output_path + name + "_model.hdf5")
    else:
        model.save(logdir + str(i) + "_" + name +"_model.hdf5")