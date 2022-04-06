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

def cwru_test(dataset_dict, model, testlogfile, args, train_hp, test_hp):

    testset = dataset_dict[test_hp][1]

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

    f = open(testlogfile, 'a')
    f.write('Accuracy {} --> {} : {:4}\n'.format(train_hp,test_hp, accuracy))
    f.close()

    plot_confusion_matrix_from_array(confusion_matrix(y_actu,model_answers),train_hp, test_hp, args)

def plot_confusion_matrix_from_array(df_confusion, train_hp, test_hp, args, title='Confusion matrix', cmap=plt.cm.gray_r):
    fig, ax = plt.subplots()
    ax.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    #fig.colorbar()
    tick_marks = np.arange(10)
    ax.set_xticks(tick_marks, ['normal','b007','b014','b021','ir007','ir014','ir021','or007','or014','or021'])
    ax.set_yticks(tick_marks, ['normal','b007','b014','b021','ir007','ir014','ir021','or007','or014','or021'])
    #plt.tight_layout()
    ax.tick_params(axis='x', labelrotation = 45)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Diagnosed')

    for (i, j), z in np.ndenumerate(df_confusion):
        ax.text(j, i, '{}'.format(z), ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    plt.savefig(os.path.join(args.log_path,'confusion_matrix_{}_to_{}.pdf'.format(train_hp,test_hp)), format = 'pdf')
