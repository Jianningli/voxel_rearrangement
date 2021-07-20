from glob import glob
import nrrd
import numpy as np
import cc3d
import skimage.morphology
import cv2

def skull_id(labels_out):
    labels_out=labels_out.reshape((1,-1))
    labels_out=labels_out[0,:]
    label=np.unique(labels_out)
    hist, bin_edges=np.histogram(labels_out,bins=label)
    hist=np.ndarray.tolist(hist)
    hist_=hist
    hist_=np.array(hist_)
    hist.sort(reverse = True)
    #print('hist',hist)
    idx=(hist_==hist[1])
    idx=idx+1-1
    idx_=np.sum(idx*label[0:len(idx)])
    print('idx',idx_)
    return idx_


def cca_processing(data):
    labels_out=cc3d.connected_components(data.astype('int32'))
    skull_label=skull_id(labels_out)
    skull=(labels_out==skull_label)
    skull=(skull+1-1).astype('float32')
    return skull

def clean_operation(pred, defect):
    pred[pred==defect]=0
    pred=pred.astype('float32')
    return pred

def morph(data):
    kernel = np.ones((9,9),np.uint8)
    mo=cv2.morphologyEx(data.astype('uint8'), cv2.MORPH_OPEN, kernel)
    return mo


if __name__ == "__main__":
    pred_dir='C:/Users/Jianning Li/Desktop/newBU/newBU/completion_results/test_Superres/implant/fix'
    defect_dir="C:/Users/Jianning Li/Desktop/newBU/newBU/00_Challenge_Dataset/test_set/evaluation_defective_skull/fix"
    save_dir='C:/Users/Jianning Li/Desktop/newBU/newBU/completion_results/test_Superres/implant_denoised/fix/'

    pred_list=glob('{}/*.nrrd'.format(pred_dir))
    defect_list=glob('{}/*.nrrd'.format(defect_dir))

    for i in range(len(pred_list)):
        pred,h=nrrd.read(pred_list[i])
        defect,h=nrrd.read(defect_list[i])
        moed=morph(pred)
        print('done...')
        ccaed=cca_processing(moed)
        print('done...')
        cleaned=clean_operation(ccaed,defect)
        print('done...')
        fname=save_dir+'implants%d'%i+'.nrrd'
        nrrd.write(fname,cleaned,h)




 