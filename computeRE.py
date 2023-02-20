
import os 
import numpy as np

def getDict(txt_path):
    file = open(txt_path, 'r')
    lines = file.readlines()
    rotate_dict = {}
    for line in lines:
        rotate_dict[line.split(':')[0]] = line.split(':')[1].strip('\n')
    return rotate_dict

def RotateErr(v1,v2):
    return abs(v1-v2)

if __name__ == '__main__':
    gt_path = 'data/Cataract_Test/Rotate_Index_Standard/UC'
    pred_path = 'output/predict/Txt/'
    save_result = 'output/predict/RE_result.txt'
    result_file = open(save_result, 'w')
    vid_list = sorted(os.listdir(gt_path))
    total_vids_RE = []
    for vid in vid_list:
        error_vid_list = []
        Error_vid = 0
        i_vid = 0
        txt_list = sorted(os.listdir(os.path.join(gt_path, vid)))
        for txt in txt_list:
            error_txt_list = []
            gt_txt_path = os.path.join(gt_path,vid,txt)
            pr_txt_path = os.path.join(pred_path,txt)
            gt_dict = getDict(gt_txt_path)
            pr_dict = getDict(pr_txt_path)
            for k,v in gt_dict.items():
                if gt_dict[k] == 'nan':
                    continue
                if pr_dict[k] == 'nan':
                    continue
                error_txt_list.append(RotateErr(float(gt_dict[k]), float(pr_dict[k])))
            txt_MeanRE = np.mean(error_txt_list)
            error_vid_list.append(txt_MeanRE)
            result_file.write(txt+'_MeanRE:'+str(txt_MeanRE)+'\n')
        vid_MeanRE = np.round(np.mean(error_vid_list),3)
        vid_StdRE = np.round(np.std(error_vid_list),3)
        result_file.write(vid+'  RE_Mean&&Std:'+str(vid_MeanRE) + '±' + str(vid_StdRE) +'\n')
        total_vids_RE.append(vid_MeanRE)
        result_file.write('**********************************************'+'\n')
    result_file.write('totalRE_Mean&&Std:' + str(np.round(np.mean(total_vids_RE),3))+'±'+ str(np.round(np.std(total_vids_RE),3)) + '\n')
    result_file.close()
    print('totalRE_Mean&&Std:' + str(np.round(np.mean(total_vids_RE),3))+'±'+ str(np.round(np.std(total_vids_RE),3)))



