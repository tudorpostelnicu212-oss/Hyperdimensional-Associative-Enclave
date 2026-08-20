#=============================HAE - Hyperdimensional Associative Enclave=============================

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from pathlib import Path
import hdc_model as HyDiCo
import hdc_model as hdc
from transformers import AutoTokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import tqdm
    
random_seed = random.randint(0, 2**32 - 1)

seed = 383904343
torch.manual_seed(random_seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_file(path):
    with open (path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

class Transformer(nn.Module):
    def __init__(self, vocab_size, context_size, d_model, n_heads, n_layers, d_ff, dropout):
        super().__init__()

        self.context_size = context_size
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_size, d_model)

        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer=layer, num_layers=n_layers)

        self.final_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

        self.apply(self.initialize_weights)

    def initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids, targets=None, last_only=False, output_start=None):
        sequence_length = token_ids.shape[1]

        position_start = self.context_size - sequence_length
        positions = torch.arange(position_start, self.context_size, device=token_ids.device)

        x = (self.token_embedding(token_ids) + self.position_embedding(positions))

        mask = torch.triu(torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=token_ids.device), diagonal=1)

        x = self.transformer(x, mask=mask)
        x = self.final_norm(x)

        if last_only:
            logits = self.output(x[:, -1])
        elif output_start is not None:
            logits = self.output(x[:, output_start:])
        else:
            logits = self.output(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))

        return logits, loss

def predict_word_transformer(model, tokens, token_to_id, id_to_token, max_new_tokens, temperature,):
    model.eval()
    model_device = next(model.parameters()).device

    token_ids = [token_to_id[token] for token in tokens]

    out = torch.tensor([token_ids], dtype=torch.long, device=model_device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            transformer_logits, _ = model(out[:, -model.context_size:])
            next_logits = transformer_logits[0, -1]

            probs = torch.softmax(next_logits / temperature, dim=-1)

            next_tok = torch.multinomial(probs, num_samples=1)
            out = torch.cat([out, next_tok.reshape(1, 1)], dim=1)

    out_lst = [id_to_token[token_id] for token_id in out[0].tolist()]

    return out_lst

def make_transformer_training_data(tokens_lst, context_win_size, token_to_id, step):
    inputs = []
    targets = []

    offset = 0
    for tokens in tokens_lst:
        token_ids = torch.tensor([token_to_id[token] for token in tokens], dtype=torch.long)

        if len(token_ids) <= context_win_size:
            continue

        num_windows = len(token_ids) - context_win_size

        targets.append(torch.arange(offset, offset + num_windows, step=step))

        inputs.append(token_ids)
        offset += len(token_ids)

    train_inputs = torch.cat(inputs)
    train_targets = torch.cat(targets)

    return train_inputs, train_targets

def get_related_params(model, data, batch_size):
    train_inputs, train_targets = data

    ordered_train_ins = torch.randperm(len(train_targets), device=train_targets.device)
    results = []
    for start in range(0, len(train_targets), batch_size):
        indices = ordered_train_ins[start:start + batch_size]

        index = train_targets[indices]
        win = train_inputs[index[:, None] + torch.arange(model.context_size + 1)]

        inputs = win[:, :-1].to(device)
        targets = win[:, -1].to(device)

        model.zero_grad(set_to_none=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            transformer_logits, _ = model(inputs, last_only=True)
            if targets.ndim == 1:
                loss = F.cross_entropy(transformer_logits.reshape(-1, model.vocab_size), targets)
            else:
                loss = F.cross_entropy(transformer_logits.reshape(-1, model.vocab_size), targets[:, -1])
        loss.backward()

        scores = []
        params = []
        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            grad_flat = param.grad.detach().float().flatten()
            param_flat = param.detach().float().flatten()

            relative_change = grad_flat.abs() / (param_flat.abs() + 1e-1)

            scores.append(relative_change)
            params.append((name, param))

        scores = torch.cat(scores)
        threshold = scores.mean() + 3 * scores.std()
        top_indices = torch.nonzero(scores > threshold).squeeze(-1)

        offset = 0
        for name, param in params:
            end = offset + param.numel()

            indices = top_indices[(top_indices >= offset) & (top_indices < end)] - offset

            if indices.numel() > 0:
                results.append({"name": name, "indices": indices})

            offset = end

        model.zero_grad(set_to_none=True)
    return results

def make_transformer_replay_training_data(tokens_lst, context_size, token_to_id):
    inputs = []
    targets = []

    half = context_size // 2
    for tokens in tokens_lst:
        token_ids = torch.tensor([token_to_id[token] for token in tokens], dtype=torch.long)

        if len(token_ids) < (context_size * 2) + 1:
            continue

        inputs.append(token_ids[half: half + context_size])
        targets.append(token_ids[context_size + 1:context_size + half + 1])

        inputs.append(token_ids[context_size:context_size * 2])
        targets.append(token_ids[context_size + half + 1:context_size * 2 + 1])

    return torch.stack(inputs), torch.stack(targets)

def replay_based_train(model, epochs, optimizer, data_mush: dict, data_mush_replayer: list, replays, token_to_id, context_win_size, HyperDimComp: hdc.Hyperdimensional_Computing_Model, HDC_encoder, HDC_model, step, data_max, batch_size):
    if len(data_mush) > data_max:
        tokens_lst = []
        for data in data_mush.values():
            tokens_lst.append(data.split())

        HDC_model, data_mush = HyperDimComp.evict_data(HDC_encoder, HDC_model, tokens_lst, 0.8, data_mush, context_win_size, 128, data_max)

    for epoch in range(epochs):
        tokens_lst = []
        for replay in range(replays):
            if not data_mush_replayer:
                break

            random_replay = random.choice(data_mush_replayer)
            tokens_lst.append(random_replay.split())

            data_mush_replayer.remove(random_replay)
        
        tokens_lst_temp = []
        #texts = [" ".join(tokens) for tokens in tokens_lst]

        results, HDC_model_new, data_mush_new = HyperDimComp.HDC_predict(HDC_encoder, HDC_model, tokens_lst, top=50, threshold=0.7, min_threshold=0.3, max_threshold=0.8, data_mush=data_mush, win_size=context_win_size, batch_size=128)

        print(len(data_mush), len(data_mush_new))

        for batch_preds in results:
            for preds in batch_preds:
                for pred in preds:
                    pred = pred.item()
                    if pred == -1:
                        continue

                    tokens_lst_temp.append(data_mush[pred].split())
        
        tokens_lst.extend(tokens_lst_temp)

        HDC_model = HDC_model_new
        data_mush = data_mush_new

        train_inputs, train_targets = make_transformer_training_data(tokens_lst, context_win_size, token_to_id, step)
        trainable_params = get_related_params(model, (train_inputs, train_targets), 128)

        train_inputs, train_targets = make_transformer_replay_training_data(tokens_lst, context_win_size, token_to_id)

        masks = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
        for info in trainable_params:
            masks[info["name"]].view(-1)[info["indices"]] = 1

        with torch.no_grad():
            for name, param in model.named_parameters():
                mask = masks[name].bool()
                frozen = ~mask

                state = optimizer.state.get(param)
                if not state:
                    continue

                if "exp_avg" in state:
                    state["exp_avg"].masked_fill_(frozen, 0)

                if "exp_avg_sq" in state:
                    state["exp_avg_sq"].masked_fill_(frozen, 0)

        model.train()
        ordered_train_ins = torch.randperm(len(train_targets), device=train_targets.device)

        total_loss = 0
        batch_num = 0
        for start in tqdm.tqdm(range(0, len(train_targets), batch_size), desc="Training Transformer Model"):
            indices = ordered_train_ins[start:start + batch_size]

            inputs = train_inputs[indices].to(device)
            targets = train_targets[indices].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                transformer_logits, _ = model(inputs, last_only=False, output_start=context_win_size // 2)

                loss = F.cross_entropy(transformer_logits.reshape(-1, model.vocab_size), targets.reshape(-1))
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    param.grad.mul_(masks[name])

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            optimizer.step()

            total_loss += loss.detach()
            batch_num += 1

        print(f"epoch = {epoch}, loss = {total_loss / batch_num}")

    return data_mush_replayer, HDC_model, data_mush

def main():
    RESPONSE_LENGTH = 10

    REPLAYS_PER_EPOCH = 2048
    INPUTS_TARGET_DIS = 256
    DATA_RESERVOIR = 200000

    TEMPERATURE = 1
    CONTEXT_WIN_SIZE = 256
    TRANSFORMER_LR = 2e-3
    VOCAB_SIZE = 13000
    TRANSFORMER_BATCH_SIZE = 256

    D_MODEL = 192
    N_HEADS = 6
    N_LAYERS = 6
    D_FF = 4 * D_MODEL
    DROPOUT = 0.1

    HDC_HYPERVECTOR_LENGTH = 5000
    HDC_BATCH_SIZE = 128

    path = Path(rf"C:\AI TEST 2\Training Data")

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=["[UNK]"], vocab_size=VOCAB_SIZE)

    file_lst = [path / file for file in sorted(os.listdir(path))]

    tokenizer.train([str(file) for file in file_lst], trainer)
    del trainer
    total_files = len([str(file) for file in file_lst])

    token_to_id = tokenizer.get_vocab()
    id_to_token = {token_id: token for token, token_id in token_to_id.items()}

    model = Transformer(vocab_size=VOCAB_SIZE, context_size=CONTEXT_WIN_SIZE, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF, dropout=DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRANSFORMER_LR, weight_decay=0, fused=True)

    HyperDimComp = HyDiCo.Hyperdimensional_Computing_Model(HDC_HYPERVECTOR_LENGTH, HDC_BATCH_SIZE, token_to_id)
    HDC_model = None
    HDC_encoder = None

    task_test_tokens = []
    for file_path in file_lst:
        data = read_file(file_path)
        encoding = tokenizer.encode(data)

        ids = encoding.ids
        test_ids = torch.tensor(ids[int(len(ids) * 0.8):], dtype=torch.long)

        task_test_tokens.append(test_ids)
    del data, encoding, ids

    data_mush = {}
    for current_file, file_data in enumerate(file_lst, start=1):
        file_data = read_file(file_data)
        encoded_tokens = tokenizer.encode(file_data).tokens
        
        data_lst = encoded_tokens[:int(len(encoded_tokens) * 0.8)]
        
        data_mush_replayer = []
        for data_index, data in enumerate(data_lst, start=1):
            win_size_temp = (CONTEXT_WIN_SIZE * 2) + 1
            distance = (data_index - 1) * (CONTEXT_WIN_SIZE + 1)
            data = data_lst[distance:distance + win_size_temp]

            if len(data) < win_size_temp:
                break

            data_mush_replayer.append(" ".join(data))

        HDC_data_index = 0 if HDC_model is None else HDC_model.weight.shape[0]
        num_classes = HDC_data_index + len(data_mush_replayer)

        class_ids = []
        info_lst = []
        for index, info in enumerate(data_mush_replayer):
            class_id = HDC_data_index + index
            class_ids.append(class_id)

            data_mush[class_id] = info
            info_lst.append(info)

        HDC_model, HDC_encoder = HyperDimComp.train_HDC(model=HDC_model, encoder=HDC_encoder, num_classes=num_classes, class_labels=class_ids, train_data=info_lst)

        print(f"Current File: {current_file}/{total_files}")

        epochs = int(len(data_mush_replayer) / REPLAYS_PER_EPOCH)
        print(f"Epochs: {epochs}")

        data_mush_replayer, HDC_model, data_mush = replay_based_train(model, epochs, optimizer, data_mush, data_mush_replayer, REPLAYS_PER_EPOCH, token_to_id, CONTEXT_WIN_SIZE, HyperDimComp, HDC_encoder, HDC_model, INPUTS_TARGET_DIS, DATA_RESERVOIR, TRANSFORMER_BATCH_SIZE)

    while True:
        tokens = tokenizer.encode(input("Prompt: "))

        output = predict_word_transformer(model, tokens, token_to_id, id_to_token, RESPONSE_LENGTH, TEMPERATURE)
        output = " ".join(output)

        print(output)

if __name__ == "__main__":
    main()
