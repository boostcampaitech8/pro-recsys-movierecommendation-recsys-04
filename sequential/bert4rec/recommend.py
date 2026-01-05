import torch


@torch.no_grad()
def predict_topk_for_users(model, user_ids, train_seqs, train_seen, topk, device):
    model.eval()

    PAD = model.PAD
    MASK = model.MASK
    max_len = model.max_len

    batch = []
    pad_mask = []
    last_pos = []

    for u in user_ids:
        seq = train_seqs[u]
        tokens = [it + 1 for it in seq]  # shift (item_idx -> token_id)

        # ✅ 핵심 수정: "마지막을 MASK로 덮기"가 아니라 "끝에 MASK를 붙이기"
        tokens = tokens + [MASK]

        # 길이 자르기 (끝이 항상 MASK가 되도록)
        tokens = tokens[-max_len:]

        # padding
        if len(tokens) < max_len:
            pad_len = max_len - len(tokens)
            tokens = [PAD] * pad_len + tokens
            lp = max_len - 1
        else:
            lp = max_len - 1

        batch.append(tokens)
        pad_mask.append([t == PAD for t in tokens])
        last_pos.append(lp)

    input_ids = torch.tensor(batch, dtype=torch.long, device=device)
    pad_mask = torch.tensor(pad_mask, dtype=torch.bool, device=device)

    logits = model(input_ids, pad_mask)  # (B, L, vocab)
    bsz = logits.size(0)
    last_logits = logits[torch.arange(bsz, device=device), torch.tensor(last_pos, device=device)]

    # special tokens mask
    last_logits[:, PAD] = -1e9
    last_logits[:, MASK] = -1e9

    # seen items mask
    for i, u in enumerate(user_ids):
        seen = train_seen.get(u, None)
        if not seen:
            continue
        idx = torch.tensor(list(seen), device=device, dtype=torch.long)
        last_logits[i, idx] = -1e9

    topk_idx = torch.topk(last_logits, k=topk, dim=1).indices  # token ids (1..num_items)
    topk_item_idx = (topk_idx - 1).clamp(min=0)  # back to item_idx (0..)
    return topk_item_idx.cpu().tolist()
