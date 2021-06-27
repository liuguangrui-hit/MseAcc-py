import pandas as pd

# root_path = '../poisoning/param_data/GRU/'
# root_path = '../poisoning/param_data/RNN/'
# root_path = '../poisoning/param_data/LSTM_20000_20/'
root_path = '../poisoning/param_data/1_un_LSTM/'
weight_list = []
for i in range(50):
    path = root_path + 'time{}_dense_bias_0.csv'.format(i + 1)

    # path = root_path + 'time{}_dense_kernel_0.csv'.format(i + 1)
    weight = pd.read_csv(path, sep=',', header=None)
    print(weight)
    weight = weight.stack()
    print(weight)
    weight_list.append(weight)

weight_df = pd.DataFrame(data=weight_list)
weight_df.to_csv('un_lstm_bias.csv', sep=',', header=None, index=False, mode='w', line_terminator='\n', encoding='utf-8')
