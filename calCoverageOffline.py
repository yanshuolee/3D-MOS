import pickle
import glob

# load pickle
with open('/home/yanshuo/Documents/Multiuav/model/GridSearch/agent.pickle', 'rb') as f:
    agent = pickle.load(f)

files = glob.glob('/home/yanshuo/Documents/Multiuav/model/GridSearch/hist/*.pickle')

for file in files:
    with open(file, 'rb') as f:
        hist = pickle.load(f)
    print(hist['coverage'][-1])
    print(hist['time'][-1])
    print(hist['steps'][-1])
    print(hist['reward'][-1])


 
    
