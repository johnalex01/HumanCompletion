import torch

action_keys = ["walking", "eating", "smoking", "discussion", "directions",
               "greeting", "phoning", "posing", "purchases", "sitting",
               "sittingdown", "takingphoto", "waiting", "walkingdog", "walkingtogether"]

dim_used_3d = [2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 21, 22, 25, 26, 27, 29, 30]
dim_repeat_22 = [9, 9, 14, 16, 19, 21]
dim_repeat_32 = [16, 24, 20, 23, 28, 31]


def denorm(Am, global_max, global_min):
    Am = (Am + 1) / 2
    Am = Am * (global_max - global_min) + global_min
    return Am


def L2NormLoss_train(gt, out):
    # print(gt.shape)
    # print(['out', out.shape])
    b, s, _ = gt.shape
    gt = gt.view(b, s, -1, 3).contiguous()
    out = out.view(b, s, -1, 3).contiguous()
    loss = torch.mean(torch.norm(gt - out, 2, dim=-1))
    return loss


def L2NormLoss_test(gt, out, frame_ids):  # (batch size,feature dim, seq len)
    batch_size = gt.shape[0]
    gt = gt[:, frame_ids, :].view(batch_size, len(frame_ids), -1, 3).contiguous()
    out = out[:, frame_ids, :].view(batch_size, len(frame_ids), -1, 3).contiguous()
    loss = torch.mean(torch.norm(gt - out, 2, dim=-1), dim=(0, 2))
    return loss
