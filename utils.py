import numpy as np
import matplotlib.pyplot as plt
import os

def plot_losses(training_proc_avg, test_proc_avg):
    # to plot learning curves
    x = np.arange(1, len(training_proc_avg)+1)
    x_2 = np.linspace(1, len(training_proc_avg)+1, len(test_proc_avg))

    fig = plt.figure()
    axs = fig.add_subplot(1, 1, 1)
    axs.plot(x, training_proc_avg, label='Training loss')
    axs.plot(x_2, test_proc_avg, label='Testing loss')
    axs.set_xlabel('Epoch no.')
    axs.set_ylabel('Average loss for epoch')
    axs.set_title('Loss as training progresses')
    axs.legend()

    if not os.path.exists('results'):
        os.makedirs('results')
    fig.savefig('./results/training_loss.png', dpi=300)

def update_lr(optimizer, lr): 
    # For updating learning rate   
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def visualize_vid(seq):
    import cv2
    # opens a cv2 window to visualize the generated video
    if seq.ndim == 4:
        #print(seq.shape)
        # for normal videos (image, channel, width, height)
        for img in seq:
            cv2.imshow('vid', img)
            cv2.waitKey(1)
    else:
        # for synthesized signal videos (no_videos, no_frames, width, height, channel)
        #print(seq.shape)
        for i, vid in enumerate(seq):
            #print(vid.shape)
            for img in vid:
                cv2.imshow('vid', img)
                cv2.waitKey(1)
            print('Loaded vid ', i)
    cv2.destroyAllWindows()

def visualize_waveform(seq):
    fig, axs = plt.subplots() # nrows, ncols = 1 by default
    for waveform in seq:
        axs.plot(waveform)

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='long')[y]