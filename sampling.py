import torch 
import torch.nn.functional as F
import heapq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def greedy_decode(model, init_tokens, steps, block_size):
    model.eval()
    decoded_tokens = init_tokens[:]
    for _ in range(steps):
        x = decoded_tokens[-block_size:] if block_size > 0 else decoded_tokens
        x = torch.tensor([x], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(x)
        # necessary for NPLM vs. RNN/Transformer decoding
        logits = logits[0]
        if logits.dim() == 3:
            logits = logits[-1]
        next_token_id = torch.argmax(logits).item()
        decoded_tokens.append(next_token_id)

    return decoded_tokens

def random_decode(model, init_tokens, steps, block_size, temperature=1.0):
    model.eval()
    decoded_tokens = init_tokens[:]
    for _ in range(steps):
        x = decoded_tokens[-block_size+1:]
        x = torch.tensor([x], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(x)
        # necessary for NPLM vs. RNN/Transformer decoding
        logits = logits[0]
        if logits.dim() == 2:
            logits = logits[-1]
        probs = F.softmax(logits / temperature, dim = -1)
        next_token_id = torch.multinomial(probs, 1).item()
        decoded_tokens.append(next_token_id)

    return decoded_tokens

class BeamCandidate():
        def __init__(self, tokens, log_prob):
            self.tokens = tokens
            self.log_prob = log_prob

        def __lt__(self, other):
            return self.log_prob < other.log_prob
            
def beam_search(model, init_tokens, steps, block_size, num_beams):

    init_candidate = BeamCandidate(init_tokens[:], 0.0)
    candidates = [init_candidate]
    model.eval()

    for _ in range(steps):
        explore = []
        for cand in candidates:
            x = cand.tokens[-block_size+1:]
            x = torch.tensor([x], dtype=torch.long, device=device)

            with torch.no_grad():
                logits = model(x)
            
            logits = logits[0]
            if logits.dim() == 2:
                logits = logits[-1]

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