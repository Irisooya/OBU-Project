import torch
import torch.nn as nn
import numpy as np
import utils
import os
import json
from eval.eval_detection import ANETdetection
from eval.eval_proposal import ANETproposal
from tqdm import tqdm

confusion_matrix = np.zeros((20, 20), dtype=np.int32)
def test(net, config, logger, test_loader, test_info, step, model_file=None):
    with torch.no_grad():
        net.eval()

        if model_file is not None:
            net.load_state_dict(torch.load(model_file))

        final_res = {}
        final_res['version'] = 'VERSION 1.3'
        final_res['results'] = {}
        final_res['external_data'] = {'used': True, 'details': 'Features from I3D Network'}
        
        num_correct = 0.
        num_total = 0.

        load_iter = iter(test_loader)

        for i in range(len(test_loader.dataset)):

            _data, _label, _, vid_name, vid_num_seg = next(load_iter)

            _data = _data.cuda()
            _label = _label.cuda()
            
            vid_num_seg = vid_num_seg[0].cpu().item()
            num_segments = _data.shape[1]

            score_act, _, feat_act, feat_bkg, features, cas_softmax = net(_data)

            feat_magnitudes_act = torch.mean(torch.norm(feat_act, dim=2), dim=1)
            feat_magnitudes_bkg = torch.mean(torch.norm(feat_bkg, dim=2), dim=1)

            label_np = _label.cpu().data.numpy()
            score_np = score_act[0].cpu().data.numpy()

            pred_np = np.zeros_like(score_np)
            pred_np[np.where(score_np < config.class_thresh)] = 0
            pred_np[np.where(score_np >= config.class_thresh)] = 1

            correct_pred = np.sum(label_np == pred_np, axis=1)

            num_correct += np.sum((correct_pred == config.num_classes).astype(np.float32))
            num_total += correct_pred.shape[0]

            feat_magnitudes = torch.norm(features, p=2, dim=2)

            feat_magnitudes = utils.minmax_norm(feat_magnitudes, max_val=feat_magnitudes_act, min_val=feat_magnitudes_bkg)
            feat_magnitudes = feat_magnitudes.repeat((config.num_classes, 1, 1)).permute(1, 2, 0)

            cas = utils.minmax_norm(cas_softmax * feat_magnitudes)

            pred = np.where(score_np >= config.class_thresh)[0]

            if len(pred) == 0:
                pred = np.array([np.argmax(score_np)])

            cas_pred = cas[0].cpu().numpy()[:, pred]
            cas_pred = np.reshape(cas_pred, (num_segments, -1, 1))

            cas_pred = utils.upgrade_resolution(cas_pred, config.scale)
            
            proposal_dict = {}

            feat_magnitudes_np = feat_magnitudes[0].cpu().data.numpy()[:, pred]
            feat_magnitudes_np = np.reshape(feat_magnitudes_np, (num_segments, -1, 1))
            feat_magnitudes_np = utils.upgrade_resolution(feat_magnitudes_np, config.scale)
            
            for i in range(len(config.act_thresh_cas)):
                cas_temp = cas_pred.copy()

                zero_location = np.where(cas_temp[:, :, 0] < config.act_thresh_cas[i])
                cas_temp[zero_location] = 0

                seg_list = []
                for c in range(len(pred)):
                    pos = np.where(cas_temp[:, c, 0] > 0)
                    seg_list.append(pos)

                proposals = utils.get_proposal_oic(seg_list, cas_temp, score_np, pred, config.scale, \
                                vid_num_seg, config.feature_fps, num_segments)

                for i in range(len(proposals)):
                    class_id = proposals[i][0][0]

                    if class_id not in proposal_dict.keys():
                        proposal_dict[class_id] = []

                    proposal_dict[class_id] += proposals[i]

            for i in range(len(config.act_thresh_magnitudes)):
                cas_temp = cas_pred.copy()

                feat_magnitudes_np_temp = feat_magnitudes_np.copy()

                zero_location = np.where(feat_magnitudes_np_temp[:, :, 0] < config.act_thresh_magnitudes[i])
                feat_magnitudes_np_temp[zero_location] = 0

                seg_list = []
                for c in range(len(pred)):
                    pos = np.where(feat_magnitudes_np_temp[:, c, 0] > 0)
                    seg_list.append(pos)

                proposals = utils.get_proposal_oic(seg_list, cas_temp, score_np, pred, config.scale, \
                                vid_num_seg, config.feature_fps, num_segments)

                for i in range(len(proposals)):
                    class_id = proposals[i][0][0]

                    if class_id not in proposal_dict.keys():
                        proposal_dict[class_id] = []

                    proposal_dict[class_id] += proposals[i]
            
            final_proposals = []
            for class_id in proposal_dict.keys():
                final_proposals.append(utils.nms(proposal_dict[class_id], 0.6))

            final_res['results'][vid_name[0]] = utils.result2json(final_proposals)

        test_acc = num_correct / num_total

        json_path = os.path.join(config.output_path, 'result.json')
        with open(json_path, 'w') as f:
            json.dump(final_res, f)
            f.close()
        # tIoU_thresh = np.linspace(0.1, 0.9, 9)
        tIoU_thresh = np.linspace(0.1, 0.7, 7)
        anet_detection = ANETdetection(config.gt_path, json_path,
                                   subset='test', tiou_thresholds=tIoU_thresh,
                                   verbose=True, check_status=False)
        # mAP, average_mAP = anet_detection.evaluate()
        #######
        anet_proposal = ANETproposal(config.gt_path, json_path,
                                   subset='test', tiou_thresholds=tIoU_thresh,
                                   verbose=True, check_status=False)
        anet_proposal.evaluate()
        ap, mAP = anet_detection.evaluate()
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt


        # 使用修改后的代码计算 TP 曲线
        # 创建 ANETdetection 实例
        # 计算 TP 曲线
        # tp_curve = anet_detection.compute_tp_curve(anet_detection.ground_truth, anet_detection.prediction)
        # # 绘制 TP 曲线
        # plt.plot(np.linspace(0, 1, len(tp_curve)), tp_curve, color='blue', linestyle='--', marker='o', linewidth=2)
        # plt.xlabel('Recall')
        # plt.ylabel('True Positive (TP) Curve')
        # plt.title('TP Curve vs. Recall')
        #plt.show()
        # plt.savefig("image_eval/AP/{AP_Curve}.png")
        # print('Mean Average Precision:', mAP)
        #############



        logger.log_value('acc/Test accuracy', test_acc, step)

        for i in range(tIoU_thresh.shape[0]):
            logger.log_value('acc/mAP@{:.1f}'.format(tIoU_thresh[i]), ap[i], step)

        # logger.log_value('Average mAP', ap, step)
        logger.log_value('acc/Average mAP[0.1:0.7]', ap[:7].mean(), step)
        logger.log_value('acc/Average mAP[0.1:0.5]', ap[:5].mean(), step)
        logger.log_value('acc/Average mAP[0.3:0.7]', ap[2:7].mean(), step)

        # test_info["step"].append(step)
        # test_info["test_acc"].append(test_acc)
        # test_info["average_mAP"].append(ap)
        test_info["step"].append(step)
        test_info["test_acc"].append(test_acc)
        test_info["average_mAP[0.1:0.7]"].append(ap[:7].mean())
        test_info["average_mAP[0.1:0.5]"].append(ap[:5].mean())
        test_info["average_mAP[0.3:0.7]"].append(ap[2:7].mean())

        for i in range(tIoU_thresh.shape[0]):
            test_info["mAP@{:.1f}".format(tIoU_thresh[i])].append(ap[i])
