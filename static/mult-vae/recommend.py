import torch
import numpy as np

@torch.no_grad()
def recommend_topk(model, train_mat, topk, device, batch_size=512):
    model.eval()
    num_users = train_mat.shape[0]
    all_recs = []

    for start in range(0, num_users, batch_size):
        end = min(start + batch_size, num_users)
        
        # 배치 단위로 Sparse -> Dense 변환 및 GPU 전송
        x = torch.from_numpy(train_mat[start:end].toarray()).float().to(device)
        
        logits, _, _ = model(x)
        scores = logits.cpu().numpy() 

        # Masking: 이미 본 아이템은 추천에서 제외
        for i, u_idx in enumerate(range(start, end)):
            seen = train_mat[u_idx].indices
            scores[i, seen] = -1e9
            
        # Top-K 추출 (배치별 병렬 처리 효과)
        topk_indices = np.argsort(scores, axis=1)[:, -topk:][:, ::-1]
        all_recs.append(topk_indices)

    return np.concatenate(all_recs, axis=0)