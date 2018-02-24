#%% cell 0
import os
import sys
# Including project path
def getProjectPath():
    if(os.getcwd() == 'C:\\Users\\BOUÂMAMAElMehdi\\documents\\visual studio 2017\\Projects\\PythonBasics\\PythonBasics'):
        return 'C:/Users/BOUÂMAMAElMehdi/documents/visual studio 2017/Projects/PythonBasics/PythonBasics'
    else:
        return os.getcwd()

sys.path.append(getProjectPath())
sys.path.append(getProjectPath() + "/Content/Managers")
sys.path.append(getProjectPath() + "/Content/Models")
sys.path.append(getProjectPath() + "/Content/Helpers")
sys.path.append(getProjectPath() + "/Content/Tests")
sys.path.append(getProjectPath() + "/Configs")

import TweetToType as ttt
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import numpy as np


def main():
    model = ttt.Tweet2Type()
    train_loss, eval_loss = model.Fit()

    # Printing loss
    fig = plt.figure()

    ax1 = fig.add_subplot(2,1,1)
    ax1.set_title('Training loss per step :')
    ax1.plot(train_loss[:, 0], train_loss[:, 1])

    ax2 = fig.add_subplot(2,1,2)
    ax2.set_title('Evaluation loss per step :')
    ax2.plot(eval_loss[:, 0], train_loss[:, 1])

    fig.savefig('loss.png')

    # Save loss in files
    np.savetxt("TrainingLoss.txt",np.array(train_loss))
    np.savetxt("EvaluationLoss.txt",np.array(eval_loss))

    # Saving the model
    model.SaveModel()
    return 0

if __name__ == '__main__':
    main()
