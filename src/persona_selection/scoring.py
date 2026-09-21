"""log P(response | prompt) by teacher forcing. The core operation of the whole project.

Walked through step by step in notebooks/0.4_score_response.ipynb; this module is the same code so
later notebooks can import it.
"""
import torch


@torch.no_grad()
def score_response(model, tokenizer, prompt: str, response: str, add_eos: bool = False):
    """Return a dict with the log-probability of `response` given `prompt`, in nats.

    Tokenizes prompt + response *together* and masks out the prompt tokens. Tokenizing the two
    separately and concatenating can produce different tokens at the boundary, which silently
    changes the score. We find the prompt length by tokenizing the prompt alone and asserting that
    it is a prefix of the joint tokenization; that holds when the prompt ends at a clean boundary
    such as `Assistant:` followed by a space-initial response.

    add_eos=True appends the tokenizer's eos token so the score includes "and then the response ends".
    """
    full_ids = tokenizer(prompt + response, return_tensors="pt")["input_ids"][0]
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
    n_prompt = len(prompt_ids)
    if not torch.equal(full_ids[:n_prompt], prompt_ids):
        raise ValueError(
            "prompt tokens changed when joined with the response (boundary merge); "
            "use a prompt ending in ':' and a response starting with a space"
        )
    if add_eos:
        full_ids = torch.cat([full_ids, torch.tensor([tokenizer.eos_token_id])])
    if len(full_ids) == n_prompt:
        raise ValueError("empty response")

    ids = full_ids[None].to(model.device)
    logits = model(ids).logits[0]                      # (T, vocab), bf16
    logprobs = torch.log_softmax(logits.float(), -1)   # do the softmax in fp32

    # The logits at position t are the prediction for token t+1, so the response tokens
    # full_ids[n_prompt:] are predicted by positions n_prompt-1 ... T-2.
    targets = ids[0, n_prompt:]
    token_logprobs = logprobs[n_prompt - 1 : -1].gather(1, targets[:, None])[:, 0]

    return {
        "logprob": token_logprobs.sum().item(),
        "n_tokens": len(targets),
        "logprob_per_token": token_logprobs.mean().item(),
        "token_logprobs": token_logprobs.cpu().tolist(),
        "tokens": [tokenizer.decode([t]) for t in targets.tolist()],
    }
