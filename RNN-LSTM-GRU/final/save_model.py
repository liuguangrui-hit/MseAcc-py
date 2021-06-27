from datetime import datetime
import os


def save_model(model, i=-1, j=-1, name='model'):
    # save the model
    output_path = "saved_models/"
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    print("[INFO] {} - Saving model ...".format(now))
    logdir = output_path + name + "/"
    # try:
    #     os.mkdir(logdir)
    # except OSError as err:
    #     # print("Creation of directory {} failed:{}".format(logdir, err))
    if i == -1:
        model.save(output_path + 'NIDS_' + name + ".hdf5")
    else:
        model.save(logdir + "{}_{}_NIDS_".format(i, j) + name + ".hdf5")
