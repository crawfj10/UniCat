import torch



def get_features(model, data_loader, normalize=True, device='cuda'):
    model.eval()
    val_img_paths = []
    val_pids = []
    val_camids = []
    val_feats = []
    print(len(data_loader))
    for n_iter, (img, pid, camid_list, camids_tensor, target_view, imgpath) in enumerate(data_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids_tensor.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            val_img_paths.extend(imgpath)
            val_feats.append(feat)
            val_camids.extend(camid_list)
            val_pids.extend(pid)
        print(n_iter)
    val_feats = torch.cat(val_feats, dim=0)
    if normalize:
        val_feats = torch.nn.functional.normalize(val_feats, dim=1, p=2)
    return val_feats, val_pids, val_camids, val_img_paths

