import pandas as pd
import numpy as np
import os
import os.path
from utils import deserialize, serialize
from saliency import create_prediction_reportcard # show_saliency_maps, stitch_saliency, 

csv_name = "outputs/master_scores.csv"
dict_name = "outputs/master_scores.dict"

patch_stats = ["patch_auroc", "patch_auprc", "patch_ap"] 
img_accs =   ["acc_topk_maj", "acc_all_max", "acc_all_maj", "acc_all_weight", "acc_all_caucus_max", "acc_all_caucus_maj"]
img_aurocs = ["auroc_topk_maj", "auroc_all_max", "auroc_all_maj", "auroc_all_weight", "auroc_all_caucus_max", "auroc_all_caucus_maj"]
img_auprcs = ["auprc_topk_maj", "auprc_all_max", "auprc_all_maj", "auprc_all_weight", "auprc_all_caucus_max", "auprc_all_caucus_maj"]
img_aps =    ["ap_topk_maj", "ap_all_max", "ap_all_maj", "ap_all_weight", "ap_all_caucus_max", "ap_all_caucus_maj"]

img_pcm_stats = ['pcm_mae', 'pcm_ssim']
img_ppm_stats = ['ppm_dice', 'ppm_jaccard', 'ppm_overlap', 'ppm_sens', 'ppm_spec', 'ppm_diff', 'ppm_scagcos', 'ppm_mae', 'ppm_f',          'ppm_e', ] 
img_ssm_stats = ['ssm_dice', 'ssm_jaccard', 'ssm_overlap', 'ssm_sens', 'ssm_spec', 'ssm_diff', 'ssm_scagcos', 'ssm_mae', 'ssm_f', 'ssm_s', 'ssm_e', 'ssm_ssim']
img_sam_stats = ['sam_dice', 'sam_jaccard', 'sam_overlap', 'sam_sens', 'sam_spec', 'sam_diff', 'sam_scagcos', 'sam_mae', 'sam_f', 'sam_s', 'sam_e', 'sam_ssim']

stat_strings = [["experiment"], patch_stats, img_accs, img_aurocs, img_auprcs, img_aps, img_pcm_stats, img_ppm_stats, img_ssm_stats, img_sam_stats]


def print_train_stats(scenario_str, filter_toggle="background", patch_size=96):
    label_dict_train = "outputs/train-controls-" + scenario_str + "-" + str(patch_size) + "-" + filter_toggle + "-labeldict.obj"
    ld_train = deserialize(label_dict_train)
    
    patch_list_train = "outputs/train-controls-" + scenario_str + "-" + str(patch_size)+ "-" + filter_toggle +"-patchlist.obj"
    pl_train = deserialize(patch_list_train)

    print("Train dataset\n"+"="*50)
    print(ld_train)
    # print("# patches:", utils.count_files(patch_root + "/" + scenario_str + "/" + patch_tail))
    print("# patches:", len(pl_train))


# def print_train_stats(scenario_str):
#     label_dict_train = "outputs/train-controls-" + scenario_str + "-96-background-labeldict.obj"
#     ld_train = utils.deserialize(label_dict_train)
#     print("Train dataset\n"+"="*50)
#     print(ld_train)
#     print("# patches:", utils.count_files(patch_root + "/" + scenario_str + "/" + patch_tail))


def report_card(scenario, model, example_1, example_0, fig_path="figs", filter_toggle="background", reduce_flag=False, patch_size=96):
	specs = "-stored_random_loading-" + str(patch_size) + "-label_inherit-bce_loss-on_controls-" + scenario + "-filtration_" + filter_toggle + "_full"
	folder = "outputs/"
	label_path = folder + "test-controls-" + scenario + "-" + str(patch_size) + "-" + filter_toggle + "-labeldict.obj"
	img_path = "/oak/stanford/groups/paragm/gautam/syncontrols/1-channel/" + scenario + "/test"
	ppmgt_path = folder + "controls-" + scenario + "-" + str(patch_size) + "-" + filter_toggle + "-ppmgts.obj"
	model_name_list = [model]
	model_predpath_list = [folder + model + specs + "_preddict.obj"]

	scores_tup = create_prediction_reportcard(scenario, specs, folder, label_path, img_path, ppmgt_path, model_name_list, model_predpath_list, example_1, example_0, fig_path, num_examples=1, reduce_flag=reduce_flag)
	return scores_tup 


# def report_card(scenario, model, example_1, example_0, fig_path=figure_path, filter_toggle="background", reduce_flag=False):
#     specs = "-stored_random_loading-96-label_inherit-bce_loss-on_controls-" + scenario + "-filtration_" + filter_toggle + "_full"
#     folder = "outputs/"
#     label_path = folder + "test-controls-" + scenario + "-96-background-labeldict.obj"
#     img_path = "/oak/stanford/groups/paragm/gautam/syncontrols/1-channel/" + scenario + "/test"
#     ppmgt_path = folder + "controls-" + scenario + "-96-background-ppmgts.obj"
#     model_name_list = [model]
#     model_predpath_list = [folder + model + specs + "_preddict.obj"]
    
#     create_prediction_reportcard(scenario, specs, folder, label_path, img_path, ppmgt_path, model_name_list, model_predpath_list, example_1, example_0, fig_path, num_examples=1, reduce_flag=reduce_flag)






def write_to_dict(scores, experiment, diction):
	diction[experiment] = scores.tolist()
	print(diction)
	serialize(diction, dict_name)
	return diction

def write_to_csv(diction, cols):
	print("Woahhh", len(diction), len(cols))
	df = pd.DataFrame.from_dict(diction, orient='index', columns=cols)
	df.to_csv(csv_name, index=False)


def log_performance(scores, experiment, overwrite_flag=False):

	if not os.path.isfile(csv_name):
		print("No score CSV file detected... generating new file named master_scores.csv!")
	else:
		print("Score CSV file detected!")

	if not os.path.isfile(dict_name):
		print("No score DICT file detected... generating new file named master_scores.dict!")
		diction = {}
	else:
		print("Score DICT file detected!")
		diction = deserialize(dict_name)

		
	if overwrite_flag == False: 
		print("Overwrite flag is False. It is also set to False by default. If want to overwrite, use overwrite_flag=True in fucntion call.")
		print("Exiting...")
		return
	elif overwrite_flag == True:
		print("Overwrite flag is True... writing to master_scores.csv!")

	cols = [item for sublist in stat_strings for item in sublist]

	# flatten the results/scores list
	score_arr = [experiment]
	for category in scores:
		for el in category: # all should be lists at this level
			if isinstance(el, float) or isinstance(el, str):
				score_arr.append(el)
			else:
				for e in el:
					if isinstance(e, float) or isinstance(e, str):
						score_arr.append(e)
					else:
						for sub_e in e:
							if isinstance(sub_e, float) or isinstance(sub_e, str):
								score_arr.append(sub_e)
							else:
								print("Warning: Results are nested too deeply! Skipping...")

	print(score_arr)
	# clean the entries
	score_arr_clean = []
	for s in score_arr:
		if isinstance(s, str):
			if "guilty_superpixels" in experiment:
				if "ppm_values-s-measure" in s or "ppm_values-ssim" in s:
					continue

			if "=" in s and "(" not in s: # accuracies
				score = str(s.split("=")[1])
				score_arr_clean.append(score)
			elif "=" in s and "(" in s: # anything with ci
				score = str(s.split("=")[1])
				score_arr_clean.append(score)
			else:
				score = str(s)
				score_arr_clean.append(score)
		else:
			score = str(s)
			score_arr_clean.append(score)

	score_arr = np.array(score_arr_clean)

	print("cleaned", len(score_arr))
	print(score_arr)

	if "VGG19" in experiment:
		print(experiment)
		if "guilty_superpixels" in experiment:
			print("hi")
			n_stats = len(cols) - len(img_sam_stats)
			print(len(cols), len(img_sam_stats), n_stats, len(score_arr))
		else:
			n_stats = len(cols) - len(img_sam_stats)

		score_arr = np.expand_dims(score_arr, 1) 
		score_arr = np.reshape(score_arr, (1, n_stats))
		nans = np.empty((1, len(img_sam_stats)))
		nans[:] = np.nan

		score_arr = np.append(score_arr, nans)
		score_arr = np.expand_dims(np.array(score_arr), 1) 
		score_arr = np.reshape(score_arr, (1, len(cols)))
		score_arr =  score_arr.flatten()

	elif "VGG_att" in experiment:
		n_stats = len(cols)
		score_arr = np.expand_dims(score_arr, 1) 
		score_arr = np.reshape(score_arr, (1, n_stats))
		score_arr =  score_arr.flatten()

	print(score_arr.shape)
	print(len(cols))


	diction = write_to_dict(score_arr, experiment, diction)
	write_to_csv(diction, cols)
	print("Writing done!")





	