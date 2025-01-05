import heapq
import torch 
import torch.nn.functional as F

class BeamCandidate():
    def __init__(self, tokens, log_prob):
        self.tokens = tokens
        self.log_prob = log_prob

    def __lt__(self, other):
        return self.log_prob < other.log_prob

def beam_search(model, 
                init_tokens, 
                steps, 
                num_beams,
                block_size):


    init_candidate = BeamCandidate(init_tokens[:], 0.0)
    candidates = [init_candidate]
    model.eval()

    for _ in range(steps):
        explore = []
        for cand in candidates:
            x = cand.tokens[-block_size+1:]
            x = torch.tensor([x], dtype=torch.long, device=device)

            with torch.no_grad():
                logits = model(x)[0]

            log_probs = F.log_softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(log_probs, num_beams)

            for i in range(num_beams):
                next_prob, next_token = topk_probs[i].item(), topk_indices[i].item()
                next_candidate = BeamCandidate(
                    tokens = cand.tokens + [next_token],
                    log_prob = cand.log_prob + next_prob
                )
                heapq.heappush(explore, next_candidate)
                if len(explore) > num_beams:
                    heapq.heappop(explore)
        candidates = explore

    best_candidate = max(candidates, key = lambda c : c.log_prob)
    return best_candidate.tokens