import sklearn.metrics
import pandas as pd
import argparse

# import openslide
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as nd
from skimage import measure
import os
import sys

from utils import str2bool
from saliency import jaccard, dice, overlap
from cam_process import parse_xml

#====================
# cam16 statistics
#====================
# def computeEvaluationMask(maskDIR, resolution, level):
#     """Computes the evaluation mask. This is only for Cam16 eval masks (test-set)
    
#     Args:
#         maskDIR:    the directory of the ground truth mask
#         resolution: Pixel resolution of the image at level 0
#         level:      The level at which the evaluation mask is made
        
#     Returns:
#         evaluation_mask
#     """
#     slide = openslide.open_slide(maskDIR)
#     dims = slide.level_dimensions[level]
#     pixelarray = np.zeros(dims[0]*dims[1], dtype='uint')
#     pixelarray = np.array(slide.read_region((0,0), level, dims))
#     distance = nd.distance_transform_edt(255 - pixelarray[:,:,0])
#     Threshold = 75/(resolution * pow(2, level) * 2) # 75µm is the equivalent size of 5 tumor cells
#     binary = distance < Threshold
#     filled_image = nd.morphology.binary_fill_holes(binary)
#     evaluation_mask = measure.label(filled_image, connectivity = 2) 
#     return evaluation_mask

    
def computeITCList(evaluation_mask, resolution, level):
    """Compute the list of labels containing Isolated Tumor Cells (ITC)
    
    Description:
        A region is considered ITC if its longest diameter is below 200µm.
        As we expanded the annotations by 75µm, the major axis of the object 
        should be less than 275µm to be considered as ITC (Each pixel is 
        0.243µm*0.243µm in level 0). Therefore the major axis of the object 
        in level 5 should be less than 275/(2^5*0.243) = 35.36 pixels.
        
    Args:
        evaluation_mask:    The evaluation mask
        resolution:         Pixel resolution of the image at level 0
        level:              The level at which the evaluation mask was made
        
    Returns:
        Isolated_Tumor_Cells: list of labels containing Isolated Tumor Cells
    """
    max_label = np.amax(evaluation_mask)    
    properties = measure.regionprops(evaluation_mask)
    Isolated_Tumor_Cells = [] 
    threshold = 275/(resolution * pow(2, level))
    for i in range(0, max_label):
        if properties[i].major_axis_length < threshold:
            Isolated_Tumor_Cells.append(i+1)
    return Isolated_Tumor_Cells


def readCSVContent(csvDIR):
    """Reads the data inside CSV file
    
    Args:
        csvDIR:    The directory including all the .csv files containing the results.
        Note that the CSV files should have the same name as the original image
        
    Returns:
        Probs:      list of the Probabilities of the detected lesions
        Xcorr:      list of X-coordinates of the lesions
        Ycorr:      list of Y-coordinates of the lesions
    """
    # Xcorr, Ycorr, Probs = ([] for i in range(3))
    # csv_lines = open(csvDIR,"r").readlines()
    # for i in range(len(csv_lines)):
    #     line = csv_lines[i]
    #     elems = line.rstrip().split(',')
    #     print(elems)
    #     Probs.append(float(elems[0]))
    #     Xcorr.append(int(elems[1]))
    #     Ycorr.append(int(elems[2]))
    df = pd.read_csv(csvDIR, header=0)
    Probs = list(df["Confidence"])
    Xcorr = list(df["X coordinate"])
    Ycorr = list(df["Y coordinate"])

    Probs = [int(el) for el in Probs]
    Xcorr = [int(el) for el in Xcorr]
    Ycorr = [int(el) for el in Ycorr]
    return Probs, Xcorr, Ycorr
    
         
def compute_FP_TP_Probs(Ycorr, Xcorr, Probs, is_tumor, evaluation_mask, Isolated_Tumor_Cells, level):
    """Generates true positive and false positive stats for the analyzed image
    
    Args:
        Probs:      list of the Probabilities of the detected lesions
        Xcorr:      list of X-coordinates of the lesions
        Ycorr:      list of Y-coordinates of the lesions
        is_tumor:   A boolean variable which is one when the case cotains tumor
        evaluation_mask:    The evaluation mask
        Isolated_Tumor_Cells: list of labels containing Isolated Tumor Cells
        level:      The level at which the evaluation mask was made
         
    Returns:
        FP_probs:   A list containing the probabilities of the false positive detections
        
        TP_probs:   A list containing the probabilities of the True positive detections
        
        NumberOfTumors: Number of Tumors in the image (excluding Isolate Tumor Cells)
        
        detection_summary:   A python dictionary object with keys that are the labels 
        of the lesions that should be detected (non-ITC tumors) and values
        that contain detection details [confidence score, X-coordinate, Y-coordinate]. 
        Lesions that are missed by the algorithm have an empty value.
        
        FP_summary:   A python dictionary object with keys that represent the 
        false positive finding number and values that contain detection 
        details [confidence score, X-coordinate, Y-coordinate]. 
    """

    max_label = np.amax(evaluation_mask)
    FP_probs = [] 
    TP_probs = np.zeros((max_label,), dtype=np.float32)
    detection_summary = {}  
    FP_summary = {}
    for i in range(1,max_label+1):
        if i not in Isolated_Tumor_Cells:
            label = 'Label ' + str(i)
            detection_summary[label] = []        
     
    FP_counter = 0       
    if (is_tumor):
        for i in range(0,len(Xcorr)):
            HittedLabel = evaluation_mask[Ycorr[i]//pow(2, level), Xcorr[i]//pow(2, level)]
            if HittedLabel == 0:
                FP_probs.append(Probs[i])
                key = 'FP ' + str(FP_counter)
                FP_summary[key] = [Probs[i], Xcorr[i], Ycorr[i]]
                FP_counter+=1
            elif HittedLabel not in Isolated_Tumor_Cells:
                if (Probs[i]>TP_probs[HittedLabel-1]):
                    label = 'Label ' + str(HittedLabel)
                    detection_summary[label] = [Probs[i], Xcorr[i], Ycorr[i]]
                    TP_probs[HittedLabel-1] = Probs[i]                                     
    else:
        for i in range(0,len(Xcorr)):
            FP_probs.append(Probs[i]) 
            key = 'FP ' + str(FP_counter)
            FP_summary[key] = [Probs[i], Xcorr[i], Ycorr[i]] 
            FP_counter+=1
            
    num_of_tumors = max_label - len(Isolated_Tumor_Cells);                             
    return FP_probs, TP_probs, num_of_tumors, detection_summary, FP_summary
 
 
def computeFROC(FROC_data):
    """Generates the data required for plotting the FROC curve
    
    Args:
        FROC_data:      Contains the list of TPs, FPs, number of tumors in each image
         
    Returns:
        total_FPs:      A list containing the average number of false positives
        per image for different thresholds
        
        total_sensitivity:  A list containig overall sensitivity of the system
        for different thresholds
    """
    
    unlisted_FPs = [item for sublist in FROC_data[1] for item in sublist]
    unlisted_TPs = [item for sublist in FROC_data[2] for item in sublist] 
    
    total_FPs, total_TPs = [], []
    all_probs = sorted(set(unlisted_FPs + unlisted_TPs))
    for Thresh in all_probs[1:]:
        total_FPs.append((np.asarray(unlisted_FPs) >= Thresh).sum())
        total_TPs.append((np.asarray(unlisted_TPs) >= Thresh).sum())    
    total_FPs.append(0)
    total_TPs.append(0)
    total_FPs = np.asarray(total_FPs)/float(len(FROC_data[0]))
    total_sensitivity = np.asarray(total_TPs)/float(sum(FROC_data[3]))      
    return  total_FPs, total_sensitivity
   
   
def plotFROC(total_FPs, total_sensitivity):
    """Plots the FROC curve
    
    Args:
        total_FPs:      A list containing the average number of false positives
        per image for different thresholds
        
        total_sensitivity:  A list containig overall sensitivity of the system
        for different thresholds
         
    Returns:
        -
    """    
    fig = plt.figure()
    plt.xlabel('Average Number of False Positives', fontsize=12)
    plt.ylabel('Metastasis detection sensitivity', fontsize=12)  
    fig.suptitle('Free response receiver operating characteristic curve', fontsize=12)
    plt.plot(total_FPs, total_sensitivity, '-', color='#000000')    
    plt.show()       
      

def run_test(mask_folder, result_folder, label_dict):
    result_file_list = []
    result_file_list += [each for each in os.listdir(result_folder) if each.endswith('.csv')]
    
    EVALUATION_MASK_LEVEL = 5 # Image level at which the evaluation is done
    L0_RESOLUTION = 0.243 # pixel resolution at level 0
    
    FROC_data = np.zeros((4, len(result_file_list)), dtype=np.object)
    FP_summary = np.zeros((2, len(result_file_list)), dtype=np.object)
    detection_summary = np.zeros((2, len(result_file_list)), dtype=np.object)
    
    caseNum = 0    
    for case in result_file_list:
        id = case[0:-4].split("Z-")[1]
        print('Evaluating Performance on image:', id)
        csvDIR = os.path.join(result_folder, case)
        Probs, Xcorr, Ycorr = readCSVContent(csvDIR)
                
        is_tumor = bool(label_dict[id]) 
        if (is_tumor):
            maskDIR = mask_folder + "/" + id + "_gt.npy"
            evaluation_mask = np.load(maskDIR).astype(int)
            # print("heyo")
            ITC_labels = computeITCList(evaluation_mask, L0_RESOLUTION, EVALUATION_MASK_LEVEL)
            # print("hey")
        else:
            evaluation_mask = 0
            ITC_labels = []
            
        FROC_data[0][caseNum] = case
        FP_summary[0][caseNum] = case
        detection_summary[0][caseNum] = case
        # print("hi")
        out = compute_FP_TP_Probs(Ycorr, Xcorr, Probs, is_tumor, evaluation_mask, ITC_labels, EVALUATION_MASK_LEVEL)
        # print("hello")
        FROC_data[1][caseNum] = out[0]
        FROC_data[2][caseNum] = out[1]
        FROC_data[3][caseNum] = out[2]
        detection_summary[1][caseNum] = out[3]
        FP_summary[1][caseNum] = out[4]
        caseNum += 1
    
    # Compute FROC curve 
    total_FPs, total_sensitivity = computeFROC(FROC_data)
    print("FROC (FPs, sens):", total_FPs, total_sensitivity)
    
    # plot FROC curve
    plotFROC(total_FPs, total_sensitivity)


#=================
# Run analysis
#=================
if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--arm', required=True, type=str, help='Either choose: eval, test')
    argument_parser.add_argument('--preds_reference_path',  required=False, type=str, help='reference CSV path')
    argument_parser.add_argument('--preds_submission_path', required=False, type=str, help='submission CSV path')
    argument_parser.add_argument('--sod_reference_path',  required=False, type=str, help='reference mask path')
    argument_parser.add_argument('--sod_submission_path', required=False, type=str, help='submission mask path')
    args = argument_parser.parse_args()

    # Cam16 eval
    #------------
    if args.arm == "test":
        mask_folder = args.sod_reference
        result_folder = args.sod_submission
        label_dict = args.label_dict
        run_test(mask_folder, result_folder, label_dict)
