import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import os

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def dummy_classification_test(testset, model, testlogfile, args):
    dataloader = DataLoader(testset, batch_size = 1)

    model.eval()
    corrects = []
    model_answers = []
    y_actu = []
    for idx, (sample, label) in enumerate(dataloader):
        sample, label = sample.to(device), label.to(device)
        out = model.forward(sample)
        _, answer = torch.max(out.data,1)

        model_answer = answer==label
        model_answer = model_answer.detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        y_actu.append(label[0])
        model_answers.append(answer.detach().cpu().numpy()[0])
        corrects.append(model_answer[0])

    accuracy = np.sum(corrects)/testset.__len__()
    print(accuracy)
    f=open(testlogfile, 'a')
    f.write('Accuracy: {}'.format(accuracy))
    f.close()


    plot_confusion_matrix_from_array(confusion_matrix(y_actu,model_answers),args)

def plot_confusion_matrix_from_array(df_confusion, args, title='Confusion matrix', cmap=plt.cm.gray_r):
    fig, ax = plt.subplots()
    ax.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    #fig.colorbar()
    tick_marks = np.arange(10)
    ax.set_xticks(tick_marks, ['1','2','3','4','5','6','7','8','9','10'])
    ax.set_yticks(tick_marks, ['1','2','3','4','5','6','7','8','9','10'])
    #plt.tight_layout()
    ax.set_ylabel('Actual')
    ax.set_xlabel('Diagnosed')

    for (i, j), z in np.ndenumerate(df_confusion):
        ax.text(j, i, '{}'.format(z), ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    plt.savefig(os.path.join(args.log_path,'confusion_matrix.pdf'), format = 'pdf')
